// Finite-temperature VMC calculations on the 1D Ising model
// written by Jonathan Moussa

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// NOTES:
// - this numerical experiment was written on a very tight schedule, hence the highly suboptimal design and organization of the code
// - I took a second pass through the code to add more comments & make it a bit more understandable given the circumstances
// - the notation here is consistent w/ the accompanying paper, "Robust Decompositions of Quantum States"
// - this implementation does not support T=0 calculations except for the pure-state quantum case because of various divergences
// - for simplicity, this implementation uses C's standard pseudorandom number generator because we aren't worried about high accuracy/reliability
// - some effort is made to reduce the memory footprint of sampling by using an unsigned char to store 8 spin states
// - because the transverse-field Ising Hamiltonian is completely real in the computational basis, the quantum variational ansatz is restricted to be real
//   ^^^ a symmetry-breaking mechanism could hypothetically cause imaginary components in a minimum-energy variational state even if there are none in the true minimum-energy state
// - this implementation uses numerical derivatives, which are substantially easier to implement than analytical derivatives but also substantially less efficient
// - some minor unnecessary complications are caused by the use of unsigned int instead of int for many variables, which I was tinkering with here for no good reason
// - future implementations of these VMC methods are unlikely to reuse any code from this software

// external prototypes to required BLAS/LAPACK functions:

// LAPACK real symmetric eigensolver
void dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);

// BLAS real dense matrix-matrix multiplication
void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

// 1. Common basic functions
//==========================

// Shannon entropy function & its analytical derivatives
double entropy(double p)
{
    if(p <= 0.0 || p >= 1.0) { return 0.0; }
    return -p*log(p);
}
double entropy_deriv(double p)
{
    return -log(p) - 1.0;
}
double entropy_deriv2(double p)
{
    return -1.0/p;
}

// entropy of a Bernoulli distribution & its analytical derivatives
double bit_entropy(double p)
{
    return entropy(p) + entropy(1.0-p);
}
double bit_entropy_deriv(double p)
{
    return entropy_deriv(p) - entropy_deriv(1.0-p);
}
double bit_entropy_deriv2(double p)
{
    return entropy_deriv2(p) + entropy_deriv2(1.0-p);
}

// parameterized probability function for the classical & pure-quantum models & its analytical derivatives
double classical_model(double param)
{
    return 0.5 + 0.5*cos(2.0*param);
}
double classical_model_deriv(double param)
{
    return -sin(2.0*param);
}
double classical_model_deriv2(double param)
{
    return -2.0*cos(2.0*param);
}

// 2. Analytical results
//======================

// exact classical free energy per site for the 1D Ising model
double classical_free_energy_exact(double field, double temperature)
{
    double b = exp(1.0/temperature);
    double c = cosh(field/temperature);
    double s = sinh(field/temperature);
    return -temperature * log(b*c + sqrt(b*b*s*s + 1.0/(b*b)));
}

// exact classical energy per site for the 1D Ising model [beta * d(free energy)/d(beta)]
double classical_energy_exact(double field, double temperature)
{
    double b = exp(1.0/temperature);
    double c = cosh(field/temperature);
    double s = sinh(field/temperature);
    double term1 = sqrt(1.0/(b*b) + b*b*s*s);
    double term2 = (-1.0/(b*b) + b*b*field*c*s + b*b*s*s)/term1;
    return -classical_free_energy_exact(field,temperature) - (b*c + b*field*s + term2)/(b*c + term1);
}

// mean-field classical energy per site for the 1D Ising model (p is probability of 0 spin outcome)
double classical_energy_mean(double p, double field, double temperature)
{
    double moment = 1.0 - 2.0*p;
    return -moment*moment + field*moment;
}

// mean-field classical free energy per site for the 1D Ising model
double classical_free_energy_mean(double p, double field, double temperature)
{
    return classical_energy_mean(p,field,temperature) - temperature*bit_entropy(p);
}

// optimize the average moment for classical mean-field calculations w/ Newton's method from a rough guess
double optimal_moment(double field, double temperature)
{
    double p_min = 0.0;
    double f_min = classical_free_energy_mean(p_min,field,temperature);
    double dp = 1.0;
    int i;
    int nscan = 99;

    // coarse parameter scan for the free-energy minimum
    for(i=1 ; i<=nscan ; i++)
    {
        double p_trial = (double)i/(double)nscan;
        double f_trial = classical_free_energy_mean(p_trial,field,temperature);
        if(f_trial < f_min)
        {
            p_min = p_trial;
            f_min = f_trial;
        }
    }

    // refine the solution
    if(p_min <= 0.0) { p_min = 1e-15; } 
    if(p_min >= 1.0) { p_min = 1.0 - 1e-15; } 
    while(fabs(dp) > 1e-14)
    {
        // analytical derivatives of the classical mean-field free energy
        double f_deriv1 = 4.0*(1.0-2.0*p_min) - 2.0*field - temperature*bit_entropy_deriv(p_min);
        double f_deriv2 = -8.0 - temperature*bit_entropy_deriv2(p_min);

        // Newton's method update of the solution
        dp = -f_deriv1/f_deriv2;

        // restrict solutions to the valid domain of p in [0,1]
        if(p_min+dp >= 0.0 && p_min+dp <= 1.0)
        { p_min += dp; }
        else if(p_min+dp < 0.0)
        { p_min = 1e-15; dp = 0.0; }
        else
        { p_min = 1.0 - 1e-15; dp = 0.0; }
    }
    return p_min;
}

// high-field 2nd-order perturbation theory for the 1D periodic transverse-field Ising model
double quantum_free_energy_perturb(double field, double temperature)
{
    double x = field/temperature;
    double free_energy = -temperature*log(2.0*cosh(x)) - (0.25/temperature + 0.125*sinh(2*x)/field)/pow(cosh(x),2);
    return free_energy;
}

// free energy of a fermionic degree of freedom
double fermion_free_energy(double energy, double temperature)
{
    double occ = 1.0/(1.0 + exp(energy/temperature));
    return energy*occ - temperature*bit_entropy(occ);
}

// total energy of 1D periodic transverse-field Ising model [adapted from https://doi.org/10.1016/0003-4916(70)90270-8]
double quantum_free_energy_finite(unsigned int n, double field, double temperature)
{
    int i;
    double free_energy = 0.0;

    // sum over all the orbitals of the Jordan-Wigner fermions
    for(i=-n ; i<=(int)n ; i++)
    {
        double k = 2.0*M_PI*(double)i/(double)(2*n+1);
        double energy = 2.0*sqrt(pow(field,2) - 2.0*field*cos(k) + 1.0);
        free_energy += -0.5*energy + fermion_free_energy(energy, temperature);
    }

    // return the average free energy per site rather than the total free energy
    return free_energy/(double)(2*n+1);
}

// total energy of an infinite system (lazy estimate from a million sites, I could be more fancy & use Richardson's extrapolation to the limit)
double quantum_free_energy_exact(double field, double temperature)
{
    // this approximation is accurate enough for the purposes of this project ...
    return quantum_free_energy_finite(1000000, field, temperature);
}

// mean-field quantum energy per site (p is probability of 0 spin outcome, theta is the angle in a Given's rotation matrix)
double quantum_energy_mean(double p, double theta, double field, double temperature)
{
    double zmoment = (1.0 - 2.0*p)*(pow(cos(theta),2) - pow(sin(theta),2));
    double xmoment = (2.0*p - 1.0)*2.0*cos(theta)*sin(theta);
    return -zmoment*zmoment + field*xmoment;
}

// optimize the average moment for the mean-field calculations w/ Newton's method from rough guess
void optimal_moments(double field, double temperature, double *p, double *theta)
{
    double p_min = 0.0, theta_min = 0.0;
    double f_min = quantum_energy_mean(p_min,theta_min,field,temperature) - temperature*bit_entropy(p_min);
    double dp = 1.0, dtheta = 1.0;
    int i, j;
    int nscan = 99;

    // coarse parameter scan for the free-energy minimum
    for(i=0 ; i<=nscan ; i++)
    {
        double p_trial = (double)i/(double)nscan;
        for(j=0 ; j<=nscan ; j++)
        {
            double theta_trial = M_PI*(double)j/(double)nscan;
            double f_trial = quantum_energy_mean(p_trial,theta_trial,field,temperature) - temperature*bit_entropy(p_trial);
            if(f_trial < f_min)
            {
                p_min = p_trial;
                theta_min = theta_trial;
                f_min = f_trial;
            }
        }
    }

    // refine the solution
    if(p_min <= 0.0) { p_min = 1e-15; } 
    if(p_min >= 1.0) { p_min = 1.0 - 1e-15; } 
    while(fabs(dp) > 1e-13 || fabs(dtheta) > 1e-13)
    {
        // analytical derivatives of the primitive quantities contained in the free-energy formula
        double zmom = (1.0 - 2.0*p_min)*(pow(cos(theta_min),2) - pow(sin(theta_min),2));
        double xmom = (2.0*p_min - 1.0)*2.0*cos(theta_min)*sin(theta_min);
        double dzmom_dp = -2.0*(pow(cos(theta_min),2) - pow(sin(theta_min),2));
        double dxmom_dp = 4.0*cos(theta_min)*sin(theta_min);
        double dzmom_dt = -(1.0 - 2.0*p_min)*4.0*sin(theta_min)*cos(theta_min);
        double dxmom_dt = (2.0*p_min - 1.0)*2.0*(pow(cos(theta_min),2) - pow(sin(theta_min),2));
        double d2zmom_dpt = 8.0*sin(theta_min)*cos(theta_min);
        double d2xmom_dpt = 4.0*(pow(cos(theta_min),2) - pow(sin(theta_min),2));
        double d2zmom_dt2 = -4.0*zmom;
        double d2xmom_dt2 = -4.0*xmom;

        // analytical derivatives of the mean-field free energy
        double df_dp = -2.0*dzmom_dp*zmom + field*dxmom_dp - temperature*bit_entropy_deriv(p_min); 
        double df_dt = -2.0*dzmom_dt*zmom + field*dxmom_dt;
        double d2f_dp2 = -2.0*pow(dzmom_dp,2) - temperature*bit_entropy_deriv2(p_min);
        double d2f_dt2 = -2.0*pow(dzmom_dt,2) - 2.0*d2zmom_dt2*zmom + field*d2xmom_dt2;
        double d2f_dpt = -2.0*d2zmom_dpt*zmom - 2.0*dzmom_dt*dzmom_dp + field*d2xmom_dpt;

        // solution update using Newton's method
        dp = -(d2f_dt2*df_dp - d2f_dpt*df_dt)/(d2f_dt2*d2f_dp2 - pow(d2f_dpt,2));
        dtheta = -(-d2f_dpt*df_dp + d2f_dp2*df_dt)/(d2f_dt2*d2f_dp2 - pow(d2f_dpt,2));

        // adjust solutions if p is pinned to its boundary on [0,1]
        if(p_min+dp >= 0.0 && p_min+dp <= 1.0)
        { p_min += dp; }
        else if(p_min+dp < 0.0)
        { p_min = 1e-15; dp = 0.0; dtheta = -df_dt/d2f_dt2; }
        else
        { p_min = 1.0 - 1e-15; dp = 0.0; dtheta = -df_dt/d2f_dt2; }
        theta_min += dtheta;
    }
    *p = p_min;
    *theta = theta_min;
}

// 3. Sampling functionality
//==========================

// sample from the Bernoulli distribution using C's built-in pseudorandom number generator
unsigned char random_bit(double p)
{
    double p0 = (double)rand()/(double)RAND_MAX;
    if(p0 <= p) return 0;
    else return 1;
}

// write to a bit history that is stored as an unsigned char array (8 bits per unsigned char)
void put_history(unsigned char bit, unsigned int index, unsigned char *history)
{
    history[index/8] = (history[index/8] & ~(1 << (index%8))) | (bit%2)*(1 << (index%8));
}

// read from a bit history that is stored as an unsigned char array (8 bits per unsigned char)
unsigned char get_history(unsigned int index, unsigned char* history)
{
    return (history[index/8] & (1 << (index%8))) >> (index%8);
}

// generate a local index from the bit history & a global index to the first bit of the local index
// WARNING: this will read memory out of bounds if global_index+num_memory-1 exceeds the upper bounds on history
unsigned int local_index(unsigned int num_memory, unsigned int global_index, unsigned char *history)
{
    unsigned int i, index = 0;
    for(i=0 ; i<num_memory ; i++)
    {
        index = (index << 1) + get_history(global_index+i, history);
    }
    return index;
}

// generate a reproducible pseudorandom sequence of samples from a local conditional probability function
void generate_samples(unsigned int num_memory, unsigned int num_samples, double *conditional_update, unsigned char *history)
{
    unsigned int local_index = 0, global_index = 0;

    // seed the reproducible pseudorandomness
    srand(1);

    // sample bits in sequence using the conditional probabilities corresponding to the previous sampled bits
    for(global_index = 0 ; global_index < num_samples ; global_index++)
    {
        unsigned char bit = random_bit(conditional_update[local_index]);
        put_history(bit, global_index, history);
        local_index = ((local_index << 1) + bit) & ((1 << num_memory) - 1);
    }
}

// pair correlation analysis for a bit sequence (produces the standard "rho" value that describes the decay of the autocorrelation function)
// NOTE: this can be done much more efficiently & thoroughly with FFTs, but this is a quick, lazy implementation
// WARNING: this will return a NaN if there is no sample variance (i.e. all 0's or all 1's)
#define MAX_LENGTH 30 // maximum length over which pair correlations are calculated
double correlation_decay(unsigned int num_buffer, unsigned int num_samples, unsigned char *history)
{
    unsigned int i, j;
    double mean = 0.0, pair_correlation[MAX_LENGTH], linear = 0.0, quadratic = 0.0;

    // accumulate all of the raw pair correlation functions
    for(i=0 ; i<MAX_LENGTH ; i++)
    {
        pair_correlation[i] = 0.0;
    }
    for(i=num_buffer ; i<num_samples ; i++)
    {
        mean += (double)get_history(i,history);
        for(j=0 ; j<MAX_LENGTH && (i+j)<num_samples ; j++)
        {
            pair_correlation[j] += (double)(get_history(i,history)*get_history(i+j,history));
        }
    }

    // normalize the pair correlations & appropriately subtract the mean
    mean /= (double)(num_samples - num_buffer);
    for(j=0 ; j<MAX_LENGTH && j<(num_samples - num_buffer) ; j++)
    {
        pair_correlation[j] /= (double)(num_samples - num_buffer - j);
        pair_correlation[j] -= mean*mean;
    }

    // simple least-squares fit to sum_i (pair[i+1] - decay*pair[i])^2 = const + linear*decay + quadratic*decay^2
    for(j=1 ; j<MAX_LENGTH && j<(num_samples - num_buffer) ; j++)
    {
        linear -= 2.0*pair_correlation[j]*pair_correlation[j-1];
        quadratic += pow(pair_correlation[j-1],2);
    }

    // return the "gamma" factor that is used to adjust standard errors for local correlations [https://en.wikipedia.org/wiki/Standard_error]
    return -linear/(2.0*quadratic);
}

// 4. Classical numerical method
//==============================

// calculate the local energy of a new spin from its local probability distribution
double classical_energy_sample(unsigned int num_local, double field, double *probability)
{
    unsigned int i;
    double energy = 0.0;
    for(i=0 ; i<num_local ; i++)
    {
        // instantaneous spin values for the new spin & the previous spin
        double moment1 = 2.0*(double)(i%2) - 1.0;
        double moment2 = 2.0*(double)((i >> 1)%2) - 1.0;

        energy += probability[i]*moment1*(field - moment2);
    }
    return energy;
}

// calculate the entropy produced by adding a new spin (Shannon entropy chain rule)
double classical_entropy_production(unsigned int num_local, double *old_probability, double *new_probability)
{
    unsigned int i;
    double entropy_produced = 0.0;

    // add new entropy
    for(i=0 ; i<num_local ; i++)
    {
        entropy_produced += entropy(new_probability[i]);
    }

    // subtract old entropy
    for(i=0 ; i<num_local/2 ; i++)
    {
        entropy_produced -= entropy(old_probability[i]);
    }

    return entropy_produced;
}

// contract a local probability distribution by conditioning on the measurement outcome on the oldest spin
void contract_probability(unsigned int num_local, unsigned char outcome, double *old_probability, double *new_probability)
{
    unsigned int i;
    double norm = 0.0;

    // extract the relevant conditional probabilities
    for(i=0 ; i<num_local/2 ; i++)
    {
        new_probability[i] = old_probability[i+outcome*num_local/2];
    }

    // renormalize the probabilities
    for(i=0 ; i<num_local/2 ; i++)
    {
        norm += new_probability[i];
    }
    norm = 1.0/norm;
    for(i=0 ; i<num_local/2 ; i++)
    {
        new_probability[i] *= norm;
    }
}

// expand a local probability distribution by applying the conditional probability process
void expand_probability(unsigned int num_local, double *conditional_probability, double *old_probability, double *new_probability)
{
    unsigned int i;
    for(i=0 ; i<num_local ; i++)
    {
        new_probability[i] = conditional_probability[i]*old_probability[i/2];
    }
}

// construct the sample-averaged free energy for a fixed set of measurement outcomes
double classical_free_energy_frozen(unsigned int num_memory, unsigned int num_buffer, unsigned int num_history,
                                    double field, double temperature, unsigned char *history, double *model, double *error_bar)
{
    unsigned int i;
    unsigned int num_local = 1 << (num_memory+1); // the local region determining the new spin, including the new spin
    double *conditional_probability = (double*)malloc(sizeof(double)*num_local);
    double *probability = (double*)malloc(sizeof(double)*num_local);
    double *reduced_probability = (double*)malloc(sizeof(double)*num_local/2);
    double free_energy = 0.0, correlation;

    // very limited error checking on having an adequate number of samples
    if(num_buffer+num_memory > num_history)
    { printf("ERROR: Insufficient # of samples to initialize sampling process\n"); exit(1); }

    // construct the conditional probabilities from the model parameters
    for(i=0 ; i<num_local/2 ; i++)
    {
        conditional_probability[2*i] = classical_model(model[i]);
        conditional_probability[2*i+1] = 1.0 - conditional_probability[2*i];
    }

    // initialize the distribution at the end of the buffer to 100% probability of the known local configuration
    for(i=0 ; i<num_local/2 ; i++)
    {
        reduced_probability[i] = 0.0;
    }
    reduced_probability[local_index(num_memory, num_buffer, history)] = 1.0;

    // loop over sampled spins
    *error_bar = 0.0;
    for(i=num_buffer ; i<num_history ; i++)
    {
        double energy, entropy;

        // expand the distribution
        expand_probability(num_local, conditional_probability, reduced_probability, probability);

        // calculate the entropy production
        entropy = classical_entropy_production(num_local, reduced_probability, probability);

        // calculate the energy contribution
        energy = classical_energy_sample(num_local, field, probability);

        // update average & moment
        free_energy += energy - temperature*entropy;
        *error_bar += pow(energy - temperature*entropy,2);

        // contract the distribution for the next spin update
        contract_probability(num_local, get_history(i,history), probability, reduced_probability);
    }

    // normalize the free energy & error bar
    free_energy /= (double)(num_history-num_buffer);
    *error_bar = sqrt((*error_bar/(double)(num_history-num_buffer) - pow(free_energy,2))/(double)(num_history-num_buffer));

    // adjust the error bar to account for correlations between measurements
    correlation = correlation_decay(num_buffer, num_history, history);
    *error_bar *= sqrt((1.0 + correlation)/(1.0 - correlation));

    free(reduced_probability);
    free(probability);
    free(conditional_probability);
    return free_energy;
}

// construct the sample-averaged free energy (objective function that we are minimizing)
double classical_free_energy(unsigned int num_memory, unsigned int num_buffer, unsigned int num_history, double field, double temperature,
                             unsigned char *history, double *model, double *error_bar)
{
    unsigned int i;
    unsigned int num_local = 1 << num_memory; // the local region determining the new spin, excluding the new spin
    double *conditional_update = (double*)malloc(sizeof(double)*num_local);
    double free_energy;

    // map from model to conditional probabilities
    for(i=0 ; i<num_local ; i++)
    {
        conditional_update[i] = classical_model(model[i]);
    }

    // generate samples from the conditional distribution
    generate_samples(num_memory, num_history, conditional_update, history);

    // calculate the fixed-sample free energy
    free_energy = classical_free_energy_frozen(num_memory, num_buffer, num_history, field, temperature, history, model, error_bar);
    free(conditional_update);
    return free_energy;
}

// numerical derivatives of the classical free energy w/ a fixed set of measurement outcomes
#define MODEL_DELTA 1e-3 // absolute variable offset used in all finite-difference calculations
double classical_free_energy_deriv(unsigned int num_memory, unsigned int num_buffer, unsigned int num_history, double field, double temperature,
                                   unsigned char *history, double *model, double *dmodel, double *d2model, double *error_bar)
{
    unsigned int i;
    unsigned int num_local = 1 << num_memory;
    double free_energy = classical_free_energy_frozen(num_memory, num_buffer, num_history, field, temperature, history, model, error_bar), free_energy_plus, free_energy_minus;

    for(i=0 ; i<num_local ; i++)
    {
        double storage;
        storage = model[i];
        model[i] += MODEL_DELTA;
        free_energy_plus = classical_free_energy_frozen(num_memory, num_buffer, num_history, field, temperature, history, model, error_bar);
        model[i] -= 2.0*MODEL_DELTA;
        free_energy_minus = classical_free_energy_frozen(num_memory, num_buffer, num_history, field, temperature, history, model, error_bar);
        model[i] = storage;

        // finite-difference formulae for 1st & 2nd derivatives
        dmodel[i] = (free_energy_plus - free_energy_minus)/(2.0*MODEL_DELTA);
        d2model[i] = (free_energy_plus + free_energy_minus - 2.0*free_energy)/(MODEL_DELTA*MODEL_DELTA);
    }

    return free_energy;
}

// optimization of the classical free energy using robust line searches (an initial guess for the model is assumed as input)
#define HESSIAN_SHIFT 1e-3
double optimize_classical_free_energy(unsigned int num_memory, unsigned int num_buffer, unsigned int num_history, double field, double temperature, double *model, double *error_bar)
{
    unsigned int i, j, min_index = 1;
    unsigned int num_local = 1 << num_memory;
    unsigned int nscan = 101;
    unsigned char *history = (unsigned char*)malloc(sizeof(unsigned char)*(num_history/8 + 1));
    double *energy_scan = (double*)malloc(sizeof(double)*nscan);
    double *model2 = (double*)malloc(sizeof(double)*num_local);
    double *dmodel = (double*)malloc(sizeof(double)*num_local);
    double *d2model = (double*)malloc(sizeof(double)*num_local);
    double *conditional_update = (double*)malloc(sizeof(double)*num_local);
    double min_energy;
    while(min_index != 0)
    {
        // map from model to conditional probabilities
        for(i=0 ; i<num_local ; i++)
        {
            conditional_update[i] = classical_model(model[i]);
        }
        // generate samples from the conditional distribution
        generate_samples(num_memory, num_history, conditional_update, history);

        // calculate a search direction
        classical_free_energy_deriv(num_memory, num_buffer, num_history, field, temperature, history, model, dmodel, d2model, error_bar);
        for(i=0 ; i<num_local ; i++)
        {
            dmodel[i] /= -(HESSIAN_SHIFT + fabs(d2model[i]));
        }

        // perform an energy scan
        for(i=0 ; i<nscan ; i++)
        {
            for(j=0 ; j<num_local ; j++)
            {
                model2[j] = model[j] + 2.0*dmodel[j]*(double)i/(double)(nscan-1);
            }
            energy_scan[i] = classical_free_energy(num_memory, num_buffer, num_history, field, temperature, history, model2, error_bar);
        }

        // find the minimum energy in the scan
        min_index = 0;
        min_energy = energy_scan[0];
        for(i=1 ; i<nscan ; i++)
        {
            if(energy_scan[i] < min_energy)
            {
                min_index = i;
                min_energy = energy_scan[i];
            }
        }
        for(i=0 ; i<num_local ; i++)
        {
            model[i] = model[i] + 2.0*dmodel[i]*(double)min_index/(double)(nscan-1);
        }
    }

    // re-run the classical free energy calculation to recalculate the error bar
    classical_free_energy(num_memory, num_buffer, num_history, field, temperature, history, model, error_bar);

    free(conditional_update);
    free(d2model);
    free(dmodel);
    free(model2);
    free(energy_scan);
    free(history);
    return min_energy;
}

// 5. Quantum numerical method
//============================

// calculate the local energy of a new spin from its local probability distribution
double quantum_energy_sample(unsigned int num_local, double field, double *density)
{
    unsigned int i;
    double energy = 0.0;

    // field contribution to the local energy, field*X_i
    for(i=0 ; i<num_local/2 ; i++)
    {
        double xmoment = (density[2*i+(2*i+1)*num_local] + density[(2*i+1)+2*i*num_local]);
        energy += field*xmoment;
    }

    // exchange contribution to the local energy, -Z_i*Z_{i-1}
    for(i=0 ; i<num_local/4 ; i++)
    {
        double zzmoment = density[4*i+4*i*num_local] - density[(4*i+1)+(4*i+1)*num_local] - density[4*i+2+(4*i+2)*num_local] + density[(4*i+3)+(4*i+3)*num_local];
        energy -= zzmoment;
    }
    return energy;
}

// calculate the entropy production in adding a new spin
double quantum_entropy_production(unsigned int num_local, double *old_density, double *new_density, double *work)
{
    int i, info, size, lwork = num_local*40;
    double entropy_produced = 0.0;
    char jobz = 'N', uplo = 'U';
    double *ev = work + lwork, *copy = work + lwork + num_local;

    // diagonalize old_density & calculate entropy from its eigenvalues
    size = (num_local*num_local)/4;
    for(i=0 ; i<size ; i++) { copy[i] = old_density[i]; }
    size = num_local/2;
    dsyev_(&jobz, &uplo, &size, copy, &size, ev, work, &lwork, &info);
    for(i=0 ; i<num_local/2 ; i++)
    {
        entropy_produced -= entropy(ev[i]);
    }

    // diagonalize new_density & calculate entropy from its eigenvalues
    size = num_local*num_local;
    for(i=0 ; i<size ; i++) { copy[i] = new_density[i]; }
    size = num_local;
    dsyev_(&jobz, &uplo, &size, copy, &size, ev, work, &lwork, &info);
    for(i=0 ; i<num_local ; i++)
    {
        entropy_produced += entropy(ev[i]);
    }

    return entropy_produced;
}

// contract a local density matrix by conditioning on the oldest spin
void contract_density(unsigned int num_local, unsigned char outcome, double *old_density, double *new_density)
{
    unsigned int i, j;
    double norm = 0.0;

    // extract the correct conditional density matrix
    for(i=0 ; i<num_local/2 ; i++)
    for(j=0 ; j<num_local/2 ; j++)
    {
        new_density[i+j*num_local/2] = old_density[i+outcome*num_local/2 + (j+outcome*num_local/2)*num_local];
    }

    // renormalize the density matrix
    for(i=0 ; i<num_local/2 ; i++)
    {
        norm += new_density[i+i*num_local/2];
    }
    norm = 1.0/norm;
    for(i=0 ; i<num_local/2 ; i++)
    for(j=0 ; j<num_local/2 ; j++)
    {
        new_density[i+j*num_local/2] *= norm;
    }
}

// expand a local density matrix by applying the conditional density process
void expand_density(unsigned int num_local, double *conditional_density, double *old_density, double *new_density)
{
    unsigned int i, j;
    for(i=0 ; i<num_local ; i++)
    for(j=0 ; j<num_local ; j++)
    {
        new_density[i+j*num_local] = conditional_density[i+j*num_local]*old_density[i/2+(j/2)*num_local/2];
    }
}

// apply the dephasing operation caused by a future reconstruction operation, conditioned on future measurement outcomes
// offset is the number of future qubits that are involved in the dephasing operation, index denotes the set of future measurement outcomes
void dephasing_operation(unsigned int num_local, unsigned int offset, unsigned int index, double *conditional_density, double *density)
{
    unsigned int i, j, k, l, size = num_local/offset;
    double norm = 0.0;

    // dephase the density matrix based on a future reconstruction operation & future qubit measurements
    for(i=0 ; i<offset ; i++)
    for(j=0 ; j<offset ; j++)
    for(k=0 ; k<size ; k++)
    for(l=0 ; l<size ; l++)
    {
        density[(l+size*j)+(k+size*i)*num_local] *= conditional_density[(index+l*offset)+(index+k*offset)*num_local];
    }

    // renormalize the dephased density matrix
    for(i=0 ; i<num_local ; i++)
    {
        norm += density[i+i*num_local];
    }
    norm = 1.0/norm;
    for(i=0 ; i<num_local ; i++)
    for(j=0 ; j<num_local ; j++)
    {
        density[i+j*num_local] *= norm;
    }
}

// construct the sample-averaged energy for a fixed set of measurement outcomes
double quantum_energy_frozen(unsigned int num_memory, unsigned int num_buffer, unsigned int num_history,
                             double field, unsigned char *history, double *model, double *error_bar)
{
    unsigned int i, j;
    unsigned int num_local = 1 << (num_memory+1); // the local region determining the new spin, including the new spin
    double *conditional_density = (double*)malloc(sizeof(double)*num_local*num_local);
    double *density = (double*)malloc(sizeof(double)*num_local*num_local);
    double *reduced_density = (double*)malloc(sizeof(double)*num_local*num_local/4);
    double *dephased_density = (double*)malloc(sizeof(double)*num_local*num_local);
    double energy = 0.0, correlation;

    // very limited error checking that there are enough samples
    if(num_buffer+2*num_memory > num_history)
    { printf("ERROR: Insufficient # of samples for dephasing\n"); exit(1); }

    // construct the conditional density operation from the model parameters
    for(i=0 ; i<num_local/2 ; i++)
    for(j=0 ; j<num_local/2 ; j++)
    {
        conditional_density[2*i+2*j*num_local] = cos(model[i])*cos(model[j]);
        conditional_density[2*i+1+2*j*num_local] = sin(model[i])*cos(model[j]);
        conditional_density[2*i+(2*j+1)*num_local] = cos(model[i])*sin(model[j]);
        conditional_density[2*i+1+(2*j+1)*num_local] = sin(model[i])*sin(model[j]);
    }

    // initialize the distribution at the end of the buffer to a single configuration
    for(i=0 ; i<num_local/2 ; i++)
    for(j=0 ; j<num_local/2 ; j++)
    {
        reduced_density[i+j*num_local/2] = 0.0;
    }
    i = local_index(num_memory, num_buffer, history);
    reduced_density[i+i*num_local/2] = 1.0;

    // loop over sampled spins
    *error_bar = 0.0;
    for(i=num_buffer ; i<(num_history - 2*num_memory) ; i++)
    {
        unsigned int index = 0;
        double energy0;

        // expand the state
        expand_density(num_local, conditional_density, reduced_density, density);

        // dephase the observed state for every future reconstruction operation that depends on a local qubit
        for(j=0 ; j<num_local*num_local ; j++)
        {
            dephased_density[j] = density[j];
        }
        for(j=0 ; j<num_memory ; j++)
        {
            index = (index << 1) + get_history(i+j+num_memory+1, history);
            dephasing_operation(num_local, 2 << j, index, conditional_density, dephased_density);
        }

        // calculate the energy contribution
        energy0 = quantum_energy_sample(num_local, field, dephased_density);

        // update average & moment
        energy += energy0;
        *error_bar += pow(energy0,2);

        // contract the distribution for the next spin update
        contract_density(num_local, get_history(i,history), density, reduced_density);
    }

    // normalize the average energy & its error bar
    energy /= (double)(num_history-2*num_memory-num_buffer);
    *error_bar = sqrt((*error_bar/(double)(num_history-2*num_memory-num_buffer) - pow(energy,2))/(double)(num_history-2*num_memory-num_buffer));

    // adjust error bar for exponentially decaying correlations between nearby qubit measurements
    correlation = correlation_decay(num_buffer, num_history, history);
    *error_bar *= sqrt((1.0 + correlation)/(1.0 - correlation));

    free(dephased_density);
    free(reduced_density);
    free(density);
    free(conditional_density);
    return energy;
}

// construct the sample-averaged free energy for a fixed set of measurement outcomes
double quantum_free_energy_frozen(unsigned int num_memory, unsigned int num_buffer, unsigned int num_history,
                                  double field, double temperature, unsigned char *history, double *model, double *error_bar)
{
    unsigned int i, j, k;
    unsigned int num_local = 1 << (num_memory+1); // the local region determining the new spin, including the new spin
    double *conditional_density = (double*)malloc(sizeof(double)*num_local*num_local);
    double *density = (double*)malloc(sizeof(double)*num_local*num_local);
    double *reduced_density = (double*)malloc(sizeof(double)*num_local*num_local/4);
    double *dephased_density = (double*)malloc(sizeof(double)*num_local*num_local);
    double *work = (double*)malloc(sizeof(double)*num_local*(num_local+41));
    double free_energy = 0.0, correlation;
    char transa = 'T', transb = 'N';
    int size = num_local;
    double zero = 0.0, one = 1.0;

    // very limited error checking that there are enough samples
    if(num_buffer+2*num_memory > num_history)
    { printf("ERROR: Insufficient # of samples for dephasing\n"); exit(1); }

    // construct the conditional density operation from the model parameters (unnormalized Cholesky factorization)
    for(i=0 ; i<num_local ; i++)
    for(j=0 ; j<num_local ; j++)
    {
        work[i+j*num_local] = 0.0;
    }
    k = 0;
    for(i=0 ; i<num_local ; i++)
    {
        for(j=0 ; j<=i ; j++)
        {
            work[j+i*num_local] = model[k++];
        }
    }
    dgemm_(&transa, &transb, &size, &size, &size, &one, work, &size, work, &size, &zero, conditional_density, &size);

    // renormalize the conditional density operation
    for(i=0 ; i<num_local/2 ; i++)
    {
        work[2*i+1] = work[2*i] = 1.0/sqrt(conditional_density[2*i+2*i*num_local]+conditional_density[2*i+1+(2*i+1)*num_local]);
    }
    for(i=0 ; i<num_local ; i++)
    for(j=0 ; j<num_local ; j++)
    {
        conditional_density[i+j*num_local] *= work[i]*work[j];
    }

    // initialize the distribution at the end of the buffer to a single configuration
    for(i=0 ; i<num_local/2 ; i++)
    for(j=0 ; j<num_local/2 ; j++)
    {
        reduced_density[i+j*num_local/2] = 0.0;
    }
    i = local_index(num_memory, num_buffer, history);
    reduced_density[i+i*num_local/2] = 1.0;

    // loop over sampled spins
    *error_bar = 0.0;
    for(i=num_buffer ; i<(num_history - 2*num_memory) ; i++)
    {
        unsigned int index = 0;
        double energy, entropy;

        // expand the state
        expand_density(num_local, conditional_density, reduced_density, density);

        // dephase the observed state for every future reconstruction operation that is conditioned on a local qubit
        for(j=0 ; j<num_local*num_local ; j++)
        {
            dephased_density[j] = density[j];
        }
        for(j=0 ; j<num_memory ; j++)
        {
            index = (index << 1) + get_history(i+j+num_memory+1, history);
            dephasing_operation(num_local, 2 << j, index, conditional_density, dephased_density);
        }

        // calculate the entropy production
        entropy = quantum_entropy_production(num_local, reduced_density, density, work);

        // calculate the energy contribution
        energy = quantum_energy_sample(num_local, field, dephased_density);

        // update average & moment
        free_energy += energy - temperature*entropy;
        *error_bar += pow(energy - temperature*entropy,2);

        // contract the distribution for the next spin update
        contract_density(num_local, get_history(i,history), density, reduced_density);
    }

    // normalize the free energy per site & its error bar
    free_energy /= (double)(num_history-2*num_memory-num_buffer);
    *error_bar = sqrt((*error_bar/(double)(num_history-2*num_memory-num_buffer) - pow(free_energy,2))/(double)(num_history-2*num_memory-num_buffer));

    // adjust the error bar to account for exponentially decaying correlations between nearby qubit measurements
    correlation = correlation_decay(num_buffer, num_history, history);
    *error_bar *= sqrt((1.0 + correlation)/(1.0 - correlation));

    free(work);
    free(dephased_density);
    free(reduced_density);
    free(density);
    free(conditional_density);
    return free_energy;
}

// construct the sample-averaged energy (objective function that we are minimizing)
double quantum_energy(unsigned int num_memory, unsigned int num_buffer, unsigned int num_history, double field,
                      unsigned char *history, double *model, double *error_bar)
{
    unsigned int i;
    unsigned int num_local = 1 << num_memory; // the local region determining the new spin, excluding the new spin
    double *conditional_update = (double*)malloc(sizeof(double)*num_local);
    double energy;

    // map from model to conditional probabilities
    for(i=0 ; i<num_local ; i++)
    {
        conditional_update[i] = classical_model(model[i]);
    }

    // generate samples from the conditional distribution
    generate_samples(num_memory, num_history, conditional_update, history);

    // calculate the fixed-sample free energy
    energy = quantum_energy_frozen(num_memory, num_buffer, num_history, field, history, model, error_bar);
    free(conditional_update);
    return energy;
}

// calculate the conditional probability of a zero qubit from a local qubit configuration in a mixed quantum model (don't assume correct normalization)
double quantum_model(unsigned int index, double *model)
{
    unsigned int i, i0 = (2*index*(2*index+1))/2, i1 = i0 + 2*index + 1, i2 = i1 + 2*index + 2;
    double norm1 = 0.0, norm2 = 0.0;

    // 2-norm of the Cholesky factor's column associated with 0 outcome probability
    for(i=i0 ; i<i1 ; i++)
    { norm1 += pow(model[i],2); }

    // 2-norm of the Cholesky factor's column associated with 1 outcome probability
    for(i=i1 ; i<i2 ; i++)
    { norm2 += pow(model[i],2); }

    return norm1/(norm1 + norm2);
}

// renormalize a mixed quantum model so that it doesn't need to be rescaled (important for good numerical derivatives)
void renormalize_model(unsigned int num_memory, double *model)
{
    unsigned int i, j, i0, i1;
    unsigned int num_local = 1 << num_memory;
    double norm;

    for(j=0 ; j<num_local ; j++)
    {
        i0 = (2*j*(2*j+1))/2;
        i1 = i0 + 4*j + 3;
        norm = 0.0;

        // 2-norm of a pair of Cholesky factor columns that should be normalized to 1
        for(i=i0 ; i<i1 ; i++)
        { norm += pow(model[i],2); }

        // renormalize the columns
        norm = 1.0/sqrt(norm);
        for(i=i0 ; i<i1 ; i++)
        { model[i] *= norm; }
    }
}

// expand a quantum T>0 model to include another qubit within the local region
void expand_model(unsigned int num_memory, double *model)
{
    unsigned int i, j, i0, i1;
    unsigned int num_local = 1 << num_memory, num_model = (2*num_local+1)*num_local, num_model2 = (4*num_local+1)*2*num_local;
    double norm;

    // initialize the new model parameters to zero
    for(i=num_model ; i<num_model2 ; i++)
    {
        model[i] = 0.0;
    }

    // copy the leading submatrix of the Cholesky factor to the top-right submatrix block
    for(j=0 ; j<2*num_local ; j++)
    {
        i0 = (j*(j+1))/2;
        i1 = ((j+2*num_local)*(j+2*num_local+1))/2;
        for(i=0 ; i<=j ; i++)
        { model[i1+i] = model[i0+i]; }
    }
}

// construct the sample-averaged energy (objective function that we are minimizing)
double quantum_free_energy(unsigned int num_memory, unsigned int num_buffer, unsigned int num_history, double field, double temperature,
                           unsigned char *history, double *model, double *error_bar)
{
    unsigned int i;
    unsigned int num_local = 1 << num_memory; // the local region determining the new spin, excluding the new spin
    double *conditional_update = (double*)malloc(sizeof(double)*num_local);
    double free_energy;

    // map from model to conditional probabilities
    for(i=0 ; i<num_local ; i++)
    {
        conditional_update[i] = quantum_model(i, model);
    }

    // generate samples from the conditional distribution
    generate_samples(num_memory, num_history, conditional_update, history);

    // calculate the fixed-sample free energy
    free_energy = quantum_free_energy_frozen(num_memory, num_buffer, num_history, field, temperature, history, model, error_bar);
    free(conditional_update);
    return free_energy;
}

// numerical derivatives of the quantum energy w/ a fixed set of measurement outcomes
double quantum_energy_deriv(unsigned int num_memory, unsigned int num_buffer, unsigned int num_history, double field,
                                   unsigned char *history, double *model, double *dmodel, double *d2model, double *error_bar)
{
    unsigned int i;
    unsigned int num_local = 1 << num_memory;
    double energy = quantum_energy_frozen(num_memory, num_buffer, num_history, field, history, model, error_bar), energy_plus, energy_minus;

    for(i=0 ; i<num_local ; i++)
    {
        double storage;
        storage = model[i];
        model[i] += MODEL_DELTA;
        energy_plus = quantum_energy_frozen(num_memory, num_buffer, num_history, field, history, model, error_bar);
        model[i] -= 2.0*MODEL_DELTA;
        energy_minus = quantum_energy_frozen(num_memory, num_buffer, num_history, field, history, model, error_bar);
        model[i] = storage;

        // finite-difference formulae for the 1st & 2nd derivatives
        dmodel[i] = (energy_plus - energy_minus)/(2.0*MODEL_DELTA);
        d2model[i] = (energy_plus + energy_minus - 2.0*energy)/(MODEL_DELTA*MODEL_DELTA);
    }

    return energy;
}

// numerical derivatives of the quantum free energy w/ a fixed set of measurement outcomes
double quantum_free_energy_deriv(unsigned int num_memory, unsigned int num_buffer, unsigned int num_history, double field, double temperature,
                                 unsigned char *history, double *model, double *dmodel, double *d2model, double *error_bar)
{
    unsigned int i;
    unsigned int num_local = 1 << num_memory, num_model = (2*num_local+1)*num_local;
    double free_energy = quantum_free_energy_frozen(num_memory, num_buffer, num_history, field, temperature, history, model, error_bar), free_energy_plus, free_energy_minus;

    for(i=0 ; i<num_model ; i++)
    {
        double storage;
        storage = model[i];
        model[i] += MODEL_DELTA;
        free_energy_plus = quantum_free_energy_frozen(num_memory, num_buffer, num_history, field, temperature, history, model, error_bar);
        model[i] -= 2.0*MODEL_DELTA;
        free_energy_minus = quantum_free_energy_frozen(num_memory, num_buffer, num_history, field, temperature, history, model, error_bar);
        model[i] = storage;

        // finite-difference formulae for the 1st & 2nd derivatives
        dmodel[i] = (free_energy_plus - free_energy_minus)/(2.0*MODEL_DELTA);
        d2model[i] = (free_energy_plus + free_energy_minus - 2.0*free_energy)/(MODEL_DELTA*MODEL_DELTA);
    }

    return free_energy;
}

// optimization of the quantum energy using robust line searches (an initial guess for the model is assumed as input)
double optimize_quantum_energy(unsigned int num_memory, unsigned int num_buffer, unsigned int num_history, double field, double *model, double *error_bar)
{
    unsigned int i, j, min_index = 1;
    unsigned int num_local = 1 << num_memory;
    unsigned int nscan = 101;
    unsigned char *history = (unsigned char*)malloc(sizeof(unsigned char)*(num_history/8+1));
    double *energy_scan = (double*)malloc(sizeof(double)*nscan);
    double *model2 = (double*)malloc(sizeof(double)*num_local);
    double *dmodel = (double*)malloc(sizeof(double)*num_local);
    double *d2model = (double*)malloc(sizeof(double)*num_local);
    double *conditional_update = (double*)malloc(sizeof(double)*num_local);
    double min_energy;
    while(min_index != 0)
    {
        // map from model to conditional probabilities
        for(i=0 ; i<num_local ; i++)
        {
            conditional_update[i] = classical_model(model[i]);
        }
        // generate samples from the conditional distribution
        generate_samples(num_memory, num_history, conditional_update, history);

printf("energy = %e\n",quantum_energy(num_memory, num_buffer, num_history, field, history, model, error_bar));

        // calculate a search direction
        quantum_energy_deriv(num_memory, num_buffer, num_history, field, history, model, dmodel, d2model, error_bar);
        for(i=0 ; i<num_local ; i++)
        {
            dmodel[i] /= -(HESSIAN_SHIFT + fabs(d2model[i]));
        }

        // perform an energy scan
        for(i=0 ; i<nscan ; i++)
        {
            for(j=0 ; j<num_local ; j++)
            {
                model2[j] = model[j] + 2.0*dmodel[j]*(double)i/(double)(nscan-1);
            }
            energy_scan[i] = quantum_energy(num_memory, num_buffer, num_history, field, history, model2, error_bar);
        }

        // find the minimum energy in the scan
        min_index = 0;
        min_energy = energy_scan[0];
        for(i=1 ; i<nscan ; i++)
        {
            if(energy_scan[i] < min_energy)
            {
                min_index = i;
                min_energy = energy_scan[i];
            }
        }
        for(i=0 ; i<num_local ; i++)
        {
            model[i] = model[i] + 2.0*dmodel[i]*(double)min_index/(double)(nscan-1);
        }
    }

    // regenerate the error bar
    quantum_energy(num_memory, num_buffer, num_history, field, history, model, error_bar);

    free(conditional_update);
    free(d2model);
    free(dmodel);
    free(model2);
    free(energy_scan);
    free(history);
    return min_energy;
}

// optimization of the quantum free energy using robust line searches (an initial guess for the model is assumed as input)
double optimize_quantum_free_energy(unsigned int num_memory, unsigned int num_buffer, unsigned int num_history, double field, double temperature, double *model, double *error_bar)
{
    unsigned int i, j, min_index = 1, max_iter = 20, num_iter = 0;
    unsigned int num_local = 1 << num_memory, num_model = (2*num_local+1)*num_local;
    unsigned int nscan = 101;
    unsigned char *history = (unsigned char*)malloc(sizeof(unsigned char)*(num_history/8 + 1));
    double *energy_scan = (double*)malloc(sizeof(double)*nscan);
    double *model2 = (double*)malloc(sizeof(double)*num_model);
    double *dmodel = (double*)malloc(sizeof(double)*num_model);
    double *d2model = (double*)malloc(sizeof(double)*num_model);
    double *conditional_update = (double*)malloc(sizeof(double)*num_local);
    double min_energy, min_reduce = 1e-6;

    while(min_index != 0)
    {
        // map from model to conditional probabilities
        for(i=0 ; i<num_local ; i++)
        {
            conditional_update[i] = quantum_model(i, model);
        }
        // generate samples from the conditional distribution
        generate_samples(num_memory, num_history, conditional_update, history);

printf("free energy = %e +/- %e\n",quantum_free_energy(num_memory, num_buffer, num_history, field, temperature, history, model, error_bar), *error_bar);

        // calculate a search direction
        quantum_free_energy_deriv(num_memory, num_buffer, num_history, field, temperature, history, model, dmodel, d2model, error_bar);
        for(i=0 ; i<num_model ; i++)
        {
            dmodel[i] /= -(HESSIAN_SHIFT + fabs(d2model[i]));
        }

        // perform an energy scan
        for(i=0 ; i<nscan ; i++)
        {
            for(j=0 ; j<num_model ; j++)
            {
                model2[j] = model[j] + 2.0*dmodel[j]*(double)i/(double)(nscan-1);
            }
            energy_scan[i] = quantum_free_energy(num_memory, num_buffer, num_history, field, temperature, history, model2, error_bar);
        }

        // find the minimum energy in the scan
        min_index = 0;
        min_energy = energy_scan[0];
        for(i=1 ; i<nscan ; i++)
        {
            if(energy_scan[i] < min_energy)
            {
                min_index = i;
                min_energy = energy_scan[i];
            }
        }
        for(i=0 ; i<num_model ; i++)
        {
            model[i] = model[i] + 2.0*dmodel[i]*(double)min_index/(double)(nscan-1);
        }
        renormalize_model(num_memory, model);

        // break stagnation
        num_iter++;
        if( (num_iter > max_iter) || ((min_energy + min_reduce) > energy_scan[0]))
        { break; }
    }

    // regenerate the error bar
    quantum_free_energy(num_memory, num_buffer, num_history, field, temperature, history, model, error_bar);

    free(conditional_update);
    free(d2model);
    free(dmodel);
    free(model2);
    free(energy_scan);
    free(history);
    return min_energy;
}

// X. Main program
//================

int main(int argc, char **argv)
{
// This code block was used to generate the data for the mixed-state quantum energy figure:
    unsigned int i, j;
    double h = 1.15, T = 1e-13;
    double p, theta;

    unsigned int num_memory = 4;
    unsigned int num_buffer = 1000;
    unsigned int num_history = 100000;
    unsigned int num_local = 1 << num_memory, num_model = (2*num_local+1)*num_local;
    unsigned char *history = (unsigned char*)malloc(sizeof(unsigned char)*(num_history/8+1));
    double *model = (double*)malloc(sizeof(double)*num_model);
    double free_energy, error_bar, free_energy2, error_bar2, free_energy3, error_bar3, d00, d01, d11, c00, c01, c11, delta, delta_bar;

    sscanf(argv[1],"%lf",&T);

    // calculate mean-field density matrix & Cholesky decomposition
    optimal_moments(h,T,&p,&theta);
    d00 = p*pow(cos(theta),2) + (1.0 - p)*pow(sin(theta),2);
    d01 = (2.0*p - 1.0)*cos(theta)*sin(theta);
    d11 = 1.0 - d00;
    c00 = sqrt(fabs(d00));
    c01 = d01/c00;
    c11 = sqrt(fabs(d11 - d01*d01/d00));
    model[0] = c00;
    model[1] = c01;
    model[2] = c11;
    expand_model(0, model);
    free_energy = optimize_quantum_free_energy(num_memory-3, num_buffer, num_history, h, T, model, &error_bar);
    expand_model(1, model);
    delta = quantum_free_energy(num_memory-2, num_buffer, num_history, h, T, history, model, &delta_bar);
    printf("free energy w/ better entropy bound: %.14e +/- %.14e -> %.14e +/- %.14e\n",free_energy,error_bar,delta,delta_bar);
    free_energy2 = optimize_quantum_free_energy(num_memory-2, num_buffer, num_history, h, T, model, &error_bar2);
    expand_model(2, model);
    delta = quantum_free_energy(num_memory-1, num_buffer, num_history, h, T, history, model, &delta_bar);
    printf("free energy w/ better entropy bound: %.14e +/- %.14e -> %.14e +/- %.14e\n",free_energy2,error_bar2,delta,delta_bar);
    free_energy3 = optimize_quantum_free_energy(num_memory-1, num_buffer, num_history, h, T, model, &error_bar3);
    expand_model(3, model);
    delta = quantum_free_energy(num_memory, num_buffer, num_history, h, T, history, model, &delta_bar);
    printf("free energy w/ better entropy bound: %.14e +/- %.14e -> %.14e +/- %.14e\n",free_energy3,error_bar3,delta,delta_bar);
    printf("%.14e %.14e %.14e %.14e %.14e %.14e %.14e %.14e %.14e %.14e\n",T, quantum_free_energy_exact(h,T), quantum_free_energy_perturb(h,T),
           quantum_energy_mean(p,theta,h,T) - T*bit_entropy(p),free_energy,error_bar,free_energy2,error_bar2,free_energy3,error_bar3);
    free(model);
    free(history);
// This code block was used to generate the data for the pure-state quantum energy figure:
/*
    unsigned int i, j;
    double h = 2.0, T = 1e-13;
    double p, theta;

    unsigned int num_memory = 3;
    unsigned int num_buffer = 1000;
    unsigned int num_history = 100000;
    unsigned int num_local = 1 << num_memory;
    double *model = (double*)malloc(sizeof(double)*num_local);
    double free_energy, error_bar, free_energy2, error_bar2, free_energy3, error_bar3, e0;

    sscanf(argv[1],"%lf",&h);
    e0 = -1.0 - 0.25*h*h;
    if(h > 1.0) { e0 = -h - 0.25/h; }
    optimal_moments(h,T,&p,&theta);
    for(i=0 ; i<num_local/4 ; i++)
    {
        model[i] = (2.0*(double)p - 1.0)*theta;
    }
    free_energy = optimize_quantum_energy(num_memory-2, num_buffer, num_history, h, model, &error_bar);
    for(i=0 ; i<num_local/4 ; i++)
    {
        model[i+num_local/4] = model[i];
    }
    free_energy2 = optimize_quantum_energy(num_memory-1, num_buffer, num_history, h, model, &error_bar2);
    for(i=0 ; i<num_local/2 ; i++)
    {
        model[i+num_local/2] = model[i];
    }
    free_energy3 = optimize_quantum_energy(num_memory, num_buffer, num_history, h, model, &error_bar3);
    printf("%.14e %.14e %.14e %.14e %.14e %.14e %.14e %.14e %.14e %.14e\n",h, quantum_free_energy_exact(h,T), e0,
           quantum_energy_mean(p,theta,h,T) - T*bit_entropy(p),free_energy,error_bar,free_energy2,error_bar2,free_energy3,error_bar3);
    free(model);
*/
// This code block was used to generate the data for the classical free energy line search figure:
/*
    unsigned int i;
    double h = 0.3, T = 5.0;
    printf("classical vs. quantum: %e %e\n",classical_free_energy_exact(h,T),quantum_free_energy_exact(h,T));
    double p = optimal_moment(h,T);
    double q, theta;
    optimal_moments(h,T,&q,&theta);
    printf("classical vs. quantum (mean): %e %e\n",classical_free_energy_mean(p,h,T),quantum_energy_mean(q,theta,h,T) - T*bit_entropy(p));
    exit(1);

    unsigned int num_memory = 1;
    unsigned int num_buffer = 100;
    unsigned int num_history = 100000;
    unsigned int num_local = 1 << num_memory;
    double *model = (double*)malloc(sizeof(double)*num_local);
    double free_energy, error_bar;

    printf("optimal p = %e\n",p);
    printf("free energy: %e %e\n",classical_free_energy_mean(p,h,T),classical_free_energy_exact(h,T));

    for(i=0 ; i<num_local ; i++)
    {
        model[i] = acos(sqrt(p));
    }
    free_energy = optimize_classical_free_energy(num_memory, num_buffer, num_history, h, T, model, &error_bar);
    printf("free energy = %e +/- %e\n",free_energy,error_bar);

    // line searches for small numbers of samples
    double *model2 = (double*)malloc(sizeof(double)*num_local);
    for(i=0 ; i<=4 ; i++)
    {
        int nhist = pow(10,i);
        unsigned char *history = (unsigned char*)malloc(sizeof(unsigned char)*(nhist/8+1));
        for(int j=0 ; j<=10000 ; j++)
        {
            double wt = (double)j/10000.0;
            for(int k=0 ; k<num_local ; k++)
            {
                model2[k] = wt*model[k] + (1.0 - wt)*acos(sqrt(p));
            }
            free_energy = classical_free_energy(1,0,nhist,h,T,history,model2,&error_bar);
            printf(" %d %e %e %e\n",nhist,wt,free_energy,error_bar);
        }
        free(history);
    }
    free(model2);

    free(model);
*/
    return 0;
}