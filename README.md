# CWGAN-GP

The CWGAN-GP is a variant of the GAN algorithm that incorporates the Wasserstein distance metric to improve training stability and generate high-quality images.

The key feature of this implementation is the integration of conditional generation, allowing the generator to generate samples conditioned on specific labels or attributes. This enables fine-grained control over the generated outputs, making the CWGAN-GP suitable for various tasks such as image synthesis, and data augmentation.

The Gradient Penalty (GP) technique is employed to enforce smoothness and ensure Lipschitz continuity in the discriminator's gradients. By penalizing the discriminator for deviating from the desired gradient norm, the CWGAN-GP achieves better training convergence and alleviates mode collapse issues commonly encountered in GAN training.

The code provides a clean and efficient implementation, leveraging the power of PyTorch for accelerated GPU computation. It includes well-structured modules for the generator and discriminator networks, along with flexible options for network architecture, loss functions, and optimization strategies.

Additionally, this repository offers comprehensive training and evaluation scripts, allowing users to easily train CWGAN-GP models on custom datasets and evaluate their performance. The training process is accompanied by detailed logging and visualizations, enabling effective monitoring of the training progress and generated sample quality.

Whether you are interested in exploring the CWGAN-GP algorithm, developing novel generative models, or applying conditional generation to your specific domain, this repository provides a solid foundation for further experimentation and research.

## Training CWGAN-GP on fashion mnist

![Training results by epoch](training_results.gif)  

## Training CWGAN-GP on mnist

![Training results by epoch](Mnist.gif)


Training notebook:

- [cwgan-gp.ipynb](cwgan-gp.ipynb) 

- [![Open In Colab](https://img.shields.io/badge/-Open%20In%20Colab-%23F9AB00?style=flat-square&logo=google-colab&logoColor=white)](https://colab.research.google.com/github/roihezkiyahu/cwgan-gp/blob/main/cwgan-gp.ipynb) 

- [![Open In Kaggle](https://img.shields.io/badge/-Kaggle%20Notebook-%2320BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/roihezkiyahu/cwgan-gp)

