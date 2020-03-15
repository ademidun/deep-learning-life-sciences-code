# deep-learning-life-sciences-code
Code Samples for the book: [Deep Learning for the Life Sciences](https://amzn.to/3audBIt)

## Installation


If using linux: `conda create --name condaenv --file requirements.conda.linux64.txt`
If using Mac OSX: `conda create --name condaenv --file requirements.conda.osx64.txt`
If using Windows: 🤷🏿‍♂️


`conda activate ./condaenv` or `source start.sh`

To run interpreter from conda environment:
`condaenv/bin/python3.7`

# Book Notes

# Deep Learning for the Life Sciences

I think that in the near future the biggest techonological improvements in society will come from advances in artificial intelligence and biology. So I've been trying to learn more about the intersection of artificial intelligence and biology. I decided to read the book, Deep Learning for the Life Sciences because it covers both topics.

If you want to become very rich or help a lot of people in the future, I recommend learning data science and biology. More specifically deep learning and genetics. I recently bought a book about deep learning and genetics.


## Introduction Life Science is Data Science
- The book begins by talking about how modern life sciences is increasingly driven by data science and algorithms
- As I have mentioned earlier, it's not so much that the algorithms we have are more sophisticated, it's that we now have access to smarter computers
- Put mroe bluntly, the computers have gotten way smarter, we on the other hadm have gotten marginally smarter at best.

## Introduction to Deep Learning
- deep learning at it's most basic level is simply a function, f() that transforms an input x, into an output, y: y = f(x) [7]
- The simplest models are linear models of the form y = Mx + b; which is essentially just a straight line [8]
- These are very limited by the fact that they can't fit most datasets. For example, height distribution of the average person would likely not fit  a linear model

- A solution to this is basically a multilayer perceptron, which is essentially just putting one linear function inside another linear function
y = M_2(B(M_1x + b_1)) + b_2
- B is called an activation function and transforms the linear function into a non-linear function

- As you put one of these functions inside another one you create what is called a multilayer perceptron
- An interesting blog post which explains the first Perceptron/Neural net, Rosenblatt's Perceptron ([blog post](https://towardsdatascience.com/rosenblatts-perceptron-the-very-first-neural-network-37a3ec09038a) (tk find paper not behind Medium paywall), [paper](tk add paper link))

### Training Models (13)
- To train the algorithm we need a loss function, L(y,y') where Y is the actual output and y' is the target value that we expected to get
- The loss function basically takes the two values to give us a value for how wrong we are
- Usually we use the Euclidean distance 
- Also does anyone know a good way of writing mathematical notation on Github? Maybe I should use latex for this review?
- For probability distribution you should use  cross entropy (really didn't understand this part)


## Chapter 4: Machine Learning For Molecules

- random search is often used for designing interesting molecules
- How can we find more efficient ways of desigining molecules?
- The first step is to transform molecules into vectors of numbers, called moelcular featurization
- This includes things like chemical descriptor vectors, 2D graph representations,
 3D electrostatic grid representations, orbital basis function representations and more
 

### What is a molecule?
- How do you know which molecules are present in a given sample?
- use a mass spectromoter to fire a bunch of electrons at the sample
- the sample becomes ionized or charged and get propelled by an electric field
- the different fragments go into different buckets based on their mass-to-charge ratio (m/q ion mass/ion charge)
- the spread of the different charged fragments is called the spectrum
- you can then use the ratio of the different fragments in each bucket to figure out which molecule you have

- molecules are dyamic, quantum entities: the atoms in a molecule are always moving (dynamic)
    - a given molecule can be described in multiple ways (quantum)

### What are Molecular bonds?

 - Covalent bonds are strong bonds formed when two atoms share electrons
 - it takes a lot of energy to break them
 - this is what actually defines a molecule, a group of atoms joined by covalent bonds
 
 - non-covalent bonds are not as strong as covalent bonds, 
 - constantly breaking and reforming, they have a huge effect on determining shape and interaction of molecules
 - some examples include hydrogen bonds, pi-stacking, salt bridges etc.
 - but non-covalent bonds are important because most drugs interact with
 biological molecules in human body through non-covalent interactions
 
 - For example water, H20, two hydrogen atoms are strongly attached to an oxygen atom using a covalent bond and
 that is what forms a waer molecule
 
 - then different water molecules are attached to other water molecules using a hydrogen bond
 - this is what makes water the universal solvent
 
 ### Chirality of Molecules
 - some molecules come in two forms that are mirror images of each other 
 - a right form "R" form and the left-form "S" form
 - Many physcial properties are identical for both  and have identical molecular graphs
 - Important because it is possible for both forms to bind to different proteins in body and prodce various effects
 - For example, in 1950s thalidomide was prescribed as a sedative for nauseau and morning sickness for pregnant women
 but only the R form is the sedative, but the S form was a teratogenic that has been shown to cause severe defects


### Featurize Molecules
- SMILES strings are a way of describing molecules using text strings
- Extended-Connectivity Fingerprints are a way of converting arbitrary length strings into a fixed-length vector
```python
import deepchem as dc
from rdkit import Chem
smiles = ['C1CCCCC`', '01cc0cc1'] # cyclohexane and dioxane
mols = [Chem.MolFromSmiles(smile) for smile in smiles]
feat = dc.feat.CircularFingerprint(size=1024)
arr = feat.featurize(mols)
# arr is a 2-by-1024 array containing the fingerprints of the two molecules
```
- Chemical fingerprints are vectors of 1s and 0s, indicating the presence or absence of a molecular feature
- algorithm works by starting to look at each atom individually, then works outwards

- another line of thinking says that use physics of the molecule's structure to describe the molecules
- this will trypicall work best for problemsthat rely on generic propertis of the molecules 
and not detailed arrangement of atoms
```python
import deepchem as dc
from rdkit import Chem
smiles = ['C1CCCCC`', '01cc0cc1'] # cyclohexane and dioxane
mols = [Chem.MolFromSmiles(smile) for smile in smiles]
feat = dc.fet.RDKitDescriptors; 
arr = feat.featurize(mols)
# arr is a 2-by-111 array containing the properties of the two molecules
```

### Graph Convolutions

- those previous examples, all required a human that thought of an algorithm that could represent the molecules
in a way that a computer could understand
- what if there was a way to feed the graph representation of a molecule into a deep learning architecture and have the
model figure out the features of the molecule
- similiar to how a deep learning model can learn about properties of an image without being supervised
- the limitation is that the calculation is based on the molecular graph, so it doesn't know anything about
the molecule's conformation
- so it works best for small, rigid molecules, Chapter 5 looks at methods for large, flexible molecules

 