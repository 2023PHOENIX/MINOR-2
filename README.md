#  Creative text generation from Images



## 1.	Introduction


Vision is a common source of inspiration for poetry and as correctly stated by someone “an image is worth thousand words”. The objects and the sentimental imprints that one perceives from an image may lead to various feelings depending on the reader. In old days there used to be concept of “situational poetry” where artists would write poems on any given situation. We aim to achieve this result through machine learning techniques. 

The possibility of creating human-level content by AI requires building a deep and multi-modal understanding model spanning vision and language boundaries. Poetry can be inspired by many things, among which vision is one of the prime sources. 

Generating poetry from image involves the task of text generation from a given image. The area of focus being - image captioning and literature creation. The generated sentences should attain poetic grammar, structures, and language styles. and the semantic content should be relevant to the visual clues discovered from the images.

Computational poetry is a challenging research area of computer science, at the cross section of computational linguistics and artificial intelligence [2]. It has been an intriguing topic as it bridges the gap between human and machine creative ability. 

There have been approaches of image descriptions like image caption and paragraph. Image descriptions aim to generate sentence to describe facts from images in human-level languages an approach one step further is to tackle a more cognitive task i.e., generation of poetic language to an image for the purpose of poetry creation. 

Dramatic progress has been achieved by supervised convolutional neural network (CNN) models on image recognition tasks.

We generate a series of poetic sentences according to the image content. Traditional poem generation methods focus on the 1-to-1 image-poem generation. In reality, given the same image, different poets create different poems conveying different intentions, because they usually attend to different image regions and inspire different overtones.[1]








### 1.1.	Purpose of the Project

The purpose of our project is to test the creative capabilities of AI. For most pictures, humans can prepare a concise description in the form of a sentence relatively easily. Such descriptions might identify the most interesting objects, what they are doing, and where this is happening.[3] The real challenge is achieving a somewhat accurate piece of poetry with the help of AI. 

Previous works neglected such the deep relationships between the image content and human poetizing. 
Millions a taken aback by awe-inspiring beauty of poetry and yet there’s just a little bit of empirical research done in developing computational models for poetry generation. 

For simple analysis we define poetry as a piece of natural language text that complies with the constraints of grammaticality, meaningfulness, and poeticness.
1. **Grammaticality**: This states that poem should be syntactically well formed. This doesn’t necessarily mean that poem should comply with all the grammatical rules in English 
language, rather poem should be governed by set of poetic grammar rules.
2. **Meaningfulness**: Poem should convey some thought or concept. For instance, a given text may be correct by rhyme and meter to be considered a good poem but if it fails to convey particular emotion or message, it should not be considered as a good poem.
3. **Poeticness**: Poem should follow certain guidelines for style, rhyme and word-stress to be considered as a good poetic text.

 
![image](https://user-images.githubusercontent.com/55507908/168334486-e7c4ed20-48c2-4d80-87b7-df60c511d0f8.png)
Figure 1: Images described as a poem by people
                                     


### 1.2.	Target Beneficiary

Generating poetry from images has application in the creative writing industry, as a spur for artists seeking inspiration for their work.
It can also be used in advertising and marketing for captioning on unlabeled images.


### 1.3.	Related Work

Traditional approaches for poetry generation include template and grammar-based method, generative summarization under constrained optimization and statistical machine translation model. By applying deep learning approaches recent years, researches about poetry generation have entered a new stage. Recurrent neural network is widely used to generate poems that can even confuse readers from telling them from poems written by human poets. 

Previous works of poem generation mainly focus on style and rhythmic qualities of poems, while recently, some works try to address conditional poem generation, inspired by the fact that many poems are generated based on visual contents, Liu et al. [7] proposed to generate poetic sentences according to a source image.

Another approach is to build a scoring procedure that evaluates the similarity between a sentence and an image. This approach is attractive, because it is symmetric: given an image (resp. sentence), one can search for the best sentence (resp. image) in a large set. This means that one can do both illustration and annotation with one method. Another attraction is the method does not need a strong syntactic model, which is represented by the prior on sentences. 

Where the scoring procedure is built around an intermediate representation, which we call the meaning of the image (resp. sentence). In effect, image and sentence are each mapped to this intermediate space, and the results are compared; similar meanings result in a high score. The advantage of doing so is that each of these maps can be adjusted discriminatively. While the meaning space could be abstract, in our implementation we use a direct representation of simple sentences as a meaning space. This allows us to exploit distributional semantics ideas to deal with out of vocabulary words.[3]





## 2.	Project Description


### 2.1.	Reference Algorithm

**•	Artificial Neural Networks**
o	A neural network is made of artificial neurons that receive and process input data. Data is passed through the input layer, the hidden layer, and the output layer.
o	A neural network process starts when input data is fed to it. Data is then processed via its layers to provide the desired output.
o	It is defined as having one input layer, one output layer, and a few (more than one) hidden layers
               
 ![image](https://user-images.githubusercontent.com/55507908/168334606-20f85c68-6320-45eb-877f-4c1ec4ca7946.png)
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Figure 2: ANN Architecture

o	For each artificial neuron, it calculates a weighted sum of all incoming inputs by multiplying the input signal with the weights and adding a bias.
o	The weighted sum will then go through a function called an activation function to decide whether it should be fired or not. The result is the output signal for the next level.
o	The non-linearity of the network is mainly contributed by the shape of the activation function.
•	Activation Functions
o	ANNs use activation functions to perform complex computations in the hidden layers and then transfer the result to the output layer. 
o	The primary purpose of AFs is to introduce non-linear properties in the neural network. They convert the linear input signals of a node into non-linear output signals to facilitate the learning of high order polynomials that go beyond one degree for deep networks.
o	Activation function decides whether the neuron should fire or not. They are usually based on threshold values. If the neuron passes the threshold, we say that the neuron is activated or fired. If the neuron is below the threshold, we say that the neuron is not activated, or disconnected or did not fire.
o	It is crucial to set up the right activation function because of the gradient vanishing or exploding issue. The selection of right activation function depends on the problem at hand.
                
 Figure 3: Activation Function neuron working
 ![image](https://user-images.githubusercontent.com/55507908/168334652-769c5257-78ec-4e08-b981-e2e5cc6d5d43.png)

**•	Convolutional Neural Networks (CNN)**

o	CNNs are analogous to traditional ANNs in that they are comprised of neurons that self-optimize through learning. Each neuron will still receive an input and perform a operation (such as a scalar product followed by a non-linear function) - the basis of countless ANNs. 

o	From the input raw image vectors to the final output of the class score, the entire of the network will still express a single perceptive score function (the weight). The last layer will contain loss functions associated with the classes, and all of the regular tips and tricks developed for traditional ANNs still apply. 

o	The only notable difference between CNNs and traditional ANNs is that CNNs are primarily used in the field of pattern recognition within images. This allows us to encode image-specific features into the architecture, making the network more suited for image-focused tasks - whilst further reducing the parameters required to set up the model.[4] 

o	One of the largest limitations of traditional forms of ANN is that they tend to struggle with the computational complexity required to compute image data. 

o	CNNs are comprised of three types of layers. These are convolutional layers, pooling layers and fully-connected layers. When these layers are stacked, a CNN architecture has been formed.

                        
![image](https://user-images.githubusercontent.com/55507908/168334708-cf7534b4-89ef-4692-b7db-ef58be84c59e.png)
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Figure 4: CNN image matrix

o	Convolutional layer plays a vital role in how CNNs operate. The layers parameters focus around the use of learnable kernels. 6 Keiron O’Shea et al. These kernels are usually small in spatial dimensionality, but spreads along the entirety of the depth of the input. When the data hits a convolutional layer, the layer convolves each filter across the spatial dimensionality of the input to produce a 2D activation map.[4]

o	Pooling layers aim to gradually reduce the dimensionality of the representation, and thus further reduce the number of parameters and the computational complexity of the model

o	The fully-connected layer contains neurons of which are directly connected to the 
neurons in the two adjacent layers, without being connected to any layers within them.


![image](https://user-images.githubusercontent.com/55507908/168334760-f5f6f4c8-ab69-4e1f-9075-4bd3b775f1ab.png)
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Figure 5: CNN Model blueprint


(Standard neural networks have limitations when working with text data. They rely on the assumption of independence among the training and test examples. After each example (data point) is processed, the entire state of the network is lost. )


**•Recurrent Neural Networks**

o	Recurrent neural networks (RNNs) are connectionist models with the ability to selectively pass information across sequence steps, while processing sequential data one element at a time. Thus, they can model input and/or output consisting of sequences of elements that are not independent. 

o	RNN architecture faces from vanish exploding gradient to tackle this we have used LSTM “long short-term memory” comes from the following intuition. Simple recurrent neural networks have long-term memory in the form of weights. The weights change slowly during training, encoding general knowledge about the data. They also have short-term memory in the form of ephemeral activations, which pass from each node to successive nodes. The LSTM model introduces an intermediate type of storage via the memory cell.

                  
                     
**•	VGG**

o	VGG stands for Visual Geometry Group; it is a standard deep Convolutional Neural Network (CNN) architecture with multiple layers. The “deep” refers to the number of layers with VGG-16 or VGG-19 consisting of 16 and 19 convolutional layers.

o	The VGG architecture is the basis of ground-breaking object recognition models. Developed as a deep neural network, the VGGNet also surpasses baselines on many tasks and datasets beyond ImageNet. Moreover, it is now still one of the most popular image recognition architectures.

o	The VGG model, or VGGNet, that supports 16 layers is also referred to as VGG16, which is a convolutional neural network model proposed by A. Zisserman and K. Simonyan from the University of Oxford.

o	The number 16 in the name VGG refers to the fact that it is 16 layers deep neural network (VGGNet). This means that VGG16 is a pretty extensive network and has a total of around 138 million parameters. Even according to modern standards, it is a huge network. However, VGGNet16 architecture’s simplicity is what makes the network more appealing. Just by looking at its architecture, it can be said that it is quite uniform.

o	There are a few convolution layers followed by a pooling layer that reduces the height and the width. If we look at the number of filters that we can use, around 64 filters are available that we can double to about 128 and then to 256 filters. In the last layers, we can use 512 filters.

                      
                      
 
![image](https://user-images.githubusercontent.com/55507908/168334804-b4a369f6-53c2-4f91-829b-127267ba9c6e.png)
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Figure 6: VGG Architecture

## 2.2.	Characteristic of Data

•	Collection for sentence-based image description and search, consisting of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. The images were chosen from six different Flickr groups, and tend not to contain any well-known people or locations, but were manually selected to depict a variety of scenes and situations. Image data is of 234 x 234 size with 3 channels (RGB).  

•	For Poem Generation we have create a dataset manually consists of different poems, rhymes and lyrics. A small size data that used for initial phases of training has numerous poems added to it many of them are from famous poets. Poems are taken from various sources like poetryfoundations.org and google. Data used in poetry has combination of many poems with different genre which can also be used for sentiment analysis from poems but here used for generating poems.


                



### 2.3.	SWOT Analysis

**o	Strength**:
As an image is one of the strongest inspirations for poetry as the dynamic nature of poetry makes it able to be constructed from any object that we pull from an image.
We also have used transfer learning; 
The use of transfer learning helps the model to generate more accurate data. We don't have to do everything on our own, we use the pre-trained model that has been already trained on large datasets and extract the features from these models and use them for our tasks.

**o	Weakness**:
Since poetry is one of the most expressive ways to use verbal language, computational generation of texts recognizable as good poems is difficult. Unlike other types of texts, both content and form contribute to the expressivity and the aesthetical value of a poem. The extent to which the two aspects are interrelated in poetry is a matter of debate

**o	Opportunities**:
Generating poetry from images has application in the creative writing industry, as a spur for artists seeking inspiration for their work.
It can improve online marketing and customer segmentation by identifying customer interests through interpreting their shared images via social media platforms.

**o	Threats**:
An incorrectly captioned image will result in a nonsensical poem being formed.

## 3. System Requirements

### 3.1.	Software Interface

•	The source code for the project is written in python programming therefore, in order to compile the source code, the system should already have a ide and python programming language.
•	The Pytorch, Keras and TensorFlow library should be installed separately.
### 3.2.	Hardware Interface

•	It is compatible only with desktops and laptops. 
•	The minimum hardware requirements for the user to have a smooth experience while running the application are: 8GB of RAM, GPU NVIDIA GTX 1050, i5 and above, standard HDD space.
