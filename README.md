# A deep learning approach for data recovery in the Soft X-Ray fusion plasma diagnostics

## Utils
To generate the synthetic curves the [`generate_gaussian_mixture_curve()`](https://github.com/nicolagulmini/RFX_HunchColab/blob/master/utils.py) method is used. It is done because a wide variety of shapes can be obtained through a mixture of gaussians. So, in this case, the gaussians should not be seen as distributions, but only a method to obtain the curves to train the model. This method returns an array with all the x points and all the y points, in one dimension (for instance `np.array([x_1, x_2, x_3, y_1, y_2, y_3])`) because this is the format to train our model. As default, the method returns 2000 points (so an array of 4000 entries: 2000 x and 2000 y) from the mixture of two gaussians: the first one is a standard gaussian, the second one is a N(4,2). The first considered point is x=-3, the sample period is 0.005. It is possible to add some noise to each point, as a N(0, 0.001) and setting `plot=True` the method shows the plot of the curve. Here some examples:

- this is the result of the default method with plot visualization `generate_gaussian_mixture_curve(plot=True)`
- 
<img src = "https://user-images.githubusercontent.com/62892813/135228081-17cc5094-64be-4c84-8f8f-ec864ff421f4.png" width = "315" height = "210">

- here the result of `generate_gaussian_mixture_curve(n_p=4000, n_g=5, mean_and_vars=[(i*5, 1) for i in range(5)], plot=True)` 

<img src = "https://user-images.githubusercontent.com/62892813/135228089-ee04c574-155d-4340-80d9-3d51aed6fa4a.png" width = "315" height = "210">

- `generate_gaussian_mixture_curve(n_p=100, n_g=2, mean_and_vars=[(i*3, 1) for i in range(2)], start=-1, sample_period=.05, awgn_dev=.01, plot=True)`

<img src = "https://user-images.githubusercontent.com/62892813/135228808-19abffda-5ff3-4107-88f6-320e61b392cd.png" width = "315" height = "210">

Here a couple of gifs with the same curves changing the noise:

<img src = "https://user-images.githubusercontent.com/62892813/135477952-1dbbe75b-6272-4e6f-af58-b305be2675d6.gif" width = "315" height = "210"><img src = "https://user-images.githubusercontent.com/62892813/135477957-550d3a46-c1ba-4c0b-b431-df579504de5e.gif" width = "315" height = "210">

## References:
### Plasma fusion
- https://theconversation.com/we-wont-have-fusion-generators-in-five-years-but-the-holy-grail-of-clean-energy-may-still-be-on-its-way-132250
- https://www.dolcevitaonline.it/energia-green-a-che-punto-siamo-con-la-fusione-nucleare/
- https://www.linkiesta.it/2021/01/fusione-nucleare-crisi-climatica-italia-enea-eurofusion-iter/
- https://www.igi.cnr.it/ricerca/magnetic-confinement-research-in-padova/la-fisica-del-tokamak-la-centrale-a-fusione/
- https://www.igi.cnr.it/news/lo-sapevi-che-anche-il-plasma-si-autorganizza/
- https://www.igi.cnr.it/ricerca/magnetic-confinement-research-in-padova/la-configurzione-rfp/ and https://www.youtube.com/watch?v=Cw_DnPdgdWk&ab_channel=ConsorzioRFX
- https://iopscience.iop.org/article/10.1088/1741-4326/abc06c/pdf
- https://www.igi.cnr.it/ricerca/magnetic-confinement-research-in-padova/rfx-mod2/
### VAE
- [Understanding Variational Autoencoders (VAEs)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
- [A Beginner's Guide to Variational Methods: Mean-Field Approximation](https://blog.evjang.com/2016/08/variational-bayes.html)
- [Variational Inference for Neural Networks](https://towardsdatascience.com/variational-inference-for-neural-networks-a4b5cf72b24)
- [Understanding disentangling in β-VAE](https://arxiv.org/pdf/1804.03599.pdf)
- [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)
- [From Autoencoder to Beta-VAE](https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html)
