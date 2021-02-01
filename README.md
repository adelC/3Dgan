# AnglePGAN
## Progressive growing of GAN

### Types of runs
There are different 'types' of runs:

- {\bf normal run} (e.g. SURFGAN_3D/scripts/example_normal_run.jb): here, you just specify all required (hyper)parameters on the command line. Optionally, use --horovod to enable data parallelism.
- {\bf run from best trial} (e.g. SURFGAN_3D/scripts/example_run_from_best_trial.jb): this run uses (hyper)parameters that were previously optimized and stored in an Optuna database. It restores an Optuna frozen trial, and runs with that. Warning: command line parameters will still take precedence if they are defined! This is intentional, in order to allow one to (partly) usehyperparameters from the frozen trial, and (partly) overwrite them with command line arguments.
- {\bf tuning using inter-trial parallelism} (e.g. SURFGAN_3D/scripts/example_hyperparam_opt_inter_trial.jb): this run aims to optimize hyperparameters using Optuna. It uses MPI to start multiple optuna trials in parallel, where each worker works on a single optuna trial. One can also continue a previous set of trials by providing the optuna_storage and optuna_study_name arguments. If these are not specified, a new trial database is create to start a fresh set of trials.
- {\bf hyperparameter tuning using intra-trial (data) parallelism} (e.g. SURFGAN_3D/scripts/example_hyperparam_opt_intra_trial.jb): this run aims to optimize hyperparameter using Optuna. In this case, a single run of the code works on a single trial: MPI is used to work on this single trial in a data-parallel fashion (much like the normal run). One can invoke multiple run's of the code on the same optuna database to nest this intra-trial parallelism with inter-trial parallelism.
