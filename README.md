# Enhanced-Surrogate-Model
It is a surrogate model designed to reasonably predict high-dimensional data such as efficiency maps.

# Purpose of development
Alright, let's think about this. Most standard deep learning algorithms take high-dimensional inputs and predict low-dimensional outputs. This is usually what you see in things like image classification. (Of course, there are models like YOLO that deal with regions, but even then, the output is still smaller than the input.)

<img width="700" height="450" alt="image" src="https://github.com/user-attachments/assets/5847ed44-b656-4c78-a18e-90c75d66146f" />

<br>
<br>
<br>

We use a surrogate model to predict things like torque ripple and efficiency based on design variables. But when designing motors for something with a wide operating range, like an EV, we need to take the efficiency map into account.

<img width="700" height="350" alt="image" src="https://github.com/user-attachments/assets/56594596-ff0f-465d-b827-16778ab9c483" />

<br>
<br>
<br>

But deep learning algorithms aren’t detectives or fortune tellers. Trying to predict high-dimensional outputs from low-dimensional inputs is really tough.

# Solutions
To address this, I propose a surrogate model based on an autoencoder.
Note: This is a basic concept developed in the early stages of the research. I later developed an improved algorithm called TD-VAE, but since it’s still under peer review, I can’t share it yet.

<br>

That was a bit of a tangent, but anyway, the key idea here is to use a model that can naturally predict high-dimensional outputs from low-dimensional inputs. And that’s exactly what an autoencoder does.
<br>
<br>
An autoencoder consists of an encoder and a decoder, with a latent space vector layer in between. When we use an autoencoder as a generative model, we usually feed random values into the latent space to generate data.
And you might be thinking, “Wait… if you think about it, we’re basically taking a low-dimensional latent vector and producing a high-dimensional output from the decoder, right?”
<br>
Exactly. In other words, by using an autoencoder, we can get high-dimensional efficiency maps (decoder outputs) from low-dimensional design variables (latent space). There are two main approaches to this. 

- First : Let’s build a neural network that can predict random latent space vectors from the design variables. This approach is very well explained in the paper "An efficient surrogate model for damage forecasting of composite laminates based on deep learning." It’s an excellent paper, and if you’re interested in surrogate models, I highly recommend giving it a read.
  <br>
- Second : This is the approach I’m proposing. Couldn’t we just train it so that the values of the latent space vectors match the original input we actually want—the design variables?
<img width="691" height="413" alt="image" src="https://github.com/user-attachments/assets/d01e5c3e-4cf7-43ce-b479-2929f1ddcded" />

# Comments:
This code demonstrates how to force the latent space vectors to learn the data that I actually want to use as input.

As I mentioned earlier, this method isn’t a perfect solution. Also, I can’t show an example applying it to a motor efficiency map yet. Stay tuned for my paper, which will be published soon, haha.
