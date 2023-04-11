import datasets
import tabular_experiment
import experiment

img_basepath = 'saved-data/plots/'

chess_experiment = tabular_experiment.TabularExperiment(datasets.TabularDataset.CHESS)
# chess_experiment.train_models(10,50,1000)
chess_experiment.load_models(10)
chess_experiment.run_experiment(experiment_type=experiment.ExperimentType.RECOURSE,n_thresholds=40,n_examples=20)
chess_experiment.visualize_hypothesis_test(experiment.ExperimentType.RECOURSE)
chess_experiment.run_experiment(experiment_type=experiment.ExperimentType.SPURIOUS,n_thresholds=40,n_examples=20)
chess_experiment.visualize_hypothesis_test(experiment.ExperimentType.SPURIOUS)

ecoli_experiment = tabular_experiment.TabularExperiment(datasets.TabularDataset.ECOLI)
# ecoli_experiment.train_models(10,100,32)
ecoli_experiment.load_models(10)
ecoli_experiment.run_experiment(experiment_type=experiment.ExperimentType.RECOURSE,n_thresholds=40,n_examples=20)
ecoli_experiment.visualize_hypothesis_test(experiment.ExperimentType.RECOURSE)
ecoli_experiment.run_experiment(experiment_type=experiment.ExperimentType.SPURIOUS,n_thresholds=40,n_examples=20)
ecoli_experiment.visualize_hypothesis_test(experiment.ExperimentType.SPURIOUS)

credit_experiment = tabular_experiment.TabularExperiment(datasets.TabularDataset.CREDIT)
# credit_experiment.train_models(10,50,64)
credit_experiment.load_models(10)
credit_experiment.run_experiment(experiment_type=experiment.ExperimentType.RECOURSE,n_thresholds=40,n_examples=20)
credit_experiment.visualize_hypothesis_test(experiment.ExperimentType.RECOURSE)
credit_experiment.run_experiment(experiment_type=experiment.ExperimentType.SPURIOUS,n_thresholds=40,n_examples=20)
credit_experiment.visualize_hypothesis_test(experiment.ExperimentType.SPURIOUS)

abalone_experiment = tabular_experiment.TabularExperiment(datasets.TabularDataset.ABALONE)
# abalone_experiment.train_models(10,20,64)
abalone_experiment.load_models(10)
abalone_experiment.run_experiment(experiment_type=experiment.ExperimentType.RECOURSE,n_thresholds=40,n_examples=20)
abalone_experiment.visualize_hypothesis_test(experiment.ExperimentType.RECOURSE)
abalone_experiment.run_experiment(experiment_type=experiment.ExperimentType.SPURIOUS,n_thresholds=40,n_examples=20)
abalone_experiment.visualize_hypothesis_test(experiment.ExperimentType.SPURIOUS)

wine_experiment = tabular_experiment.TabularExperiment(datasets.TabularDataset.WINE)
# wine_experiment.train_models(10,50,16)
wine_experiment.load_models(10)
wine_experiment.run_experiment(experiment_type=experiment.ExperimentType.RECOURSE,n_thresholds=40,n_examples=20)
wine_experiment.visualize_hypothesis_test(experiment.ExperimentType.RECOURSE)
wine_experiment.run_experiment(experiment_type=experiment.ExperimentType.SPURIOUS,n_thresholds=40,n_examples=20)
wine_experiment.visualize_hypothesis_test(experiment.ExperimentType.SPURIOUS)
