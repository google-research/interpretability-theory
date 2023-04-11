import datasets
import image_experiment
import experiment

img_basepath = 'saved-data/plots/'

cifar10_experiment = image_experiment.ImageExperiment(datasets.ImageDataset.CIFAR10)
# cifar10_experiment.train_models(10,epochs=10,batch_size=64)
cifar10_experiment.load_models(10)
cifar10_experiment.run_experiment(experiment_type=experiment.ExperimentType.RECOURSE,n_thresholds=40,n_examples=20,n_pixels=10)
cifar10_experiment.visualize_hypothesis_test(experiment.ExperimentType.RECOURSE)
cifar10_experiment.run_experiment(experiment_type=experiment.ExperimentType.SPURIOUS,n_thresholds=40,n_examples=20,n_pixels=10)
cifar10_experiment.visualize_hypothesis_test(experiment.ExperimentType.SPURIOUS)


mnist_experiment = image_experiment.ImageExperiment(datasets.ImageDataset.MNIST)
# mnist_experiment.train_models(10,epochs=10,batch_size=128)
mnist_experiment.load_models(10)
mnist_experiment.run_experiment(experiment_type=experiment.ExperimentType.RECOURSE,n_thresholds=40,n_examples=20,n_pixels=10)
mnist_experiment.visualize_hypothesis_test(experiment.ExperimentType.RECOURSE)
mnist_experiment.run_experiment(experiment_type=experiment.ExperimentType.SPURIOUS,n_thresholds=40,n_examples=20,n_pixels=10)
mnist_experiment.visualize_hypothesis_test(experiment.ExperimentType.SPURIOUS)


fashion_experiment = image_experiment.ImageExperiment(datasets.ImageDataset.FASHION)
# fashion_experiment.train_models(10,epochs=10,batch_size=128)
fashion_experiment.load_models(10)
fashion_experiment.run_experiment(experiment_type=experiment.ExperimentType.RECOURSE,n_thresholds=40,n_examples=20,n_pixels=10)
fashion_experiment.visualize_hypothesis_test(experiment.ExperimentType.RECOURSE)
fashion_experiment.run_experiment(experiment_type=experiment.ExperimentType.SPURIOUS,n_thresholds=40,n_examples=20,n_pixels=10)
fashion_experiment.visualize_hypothesis_test(experiment.ExperimentType.SPURIOUS)

