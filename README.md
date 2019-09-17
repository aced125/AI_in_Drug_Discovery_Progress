#  AI in Drug Discovery Progress

This repository contains an up-to-date list (as of September 2019) of progress (papers, github repos etc) made in applying AI to drug discovery. 


# 2019

## General
|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-----------------------------|-----------------|---|
|Brown  |`BenevolentAI`      |<a href="https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839">GuacaMol: Benchmarking Models for de Novo Molecular Design</a>       | 13 |<a href="https://github.com/BenevolentAI/guacamol">Yes</a>
|Krenn  |`Aspuru-Guzik`      |[SELFIES: a robust representation of semantically constrained graphs with an example application in chemistry](https://arxiv.org/abs/1905.13741)  | - |[Yes](https://github.com/aspuru-guzik-group/selfies)

### Brown, Fiscato, Segler, Vaucher
#### <a href="https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839">GuacaMol: Benchmarking Models for de Novo Molecular Design</a> 
>A set of benchmarks used to assess the quality of generative models. Benchmarks divided into distribution-learning, goal-directed, and assessment of compound quality.

### Krenn, Hase, Nigam, Friederich, Aspuru-Guzik
#### [SELFIES: a robust representation of semantically constrained graphs with an example application in chemistry](https://arxiv.org/abs/1905.13741)
>Alternative representation to SMILES, which seems to perform better at reconstruction accuracy in generative tests.

## Generative

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-----------------------------|-----------------|---|
|Zhavoronkov|`Insilico`     |<a href="https://www.nature.com/articles/s41587-019-0224-x">Deep learning enables the rapid identification of potent DDR1 kinase inhibitors</a>           |1|<a href="https://github.com/insilicomedicine/GENTRL">Yes</a>|
|Jensen     |`Jensen`            |<a href="https://pubs.rsc.org/en/content/articlelanding/2019/SC/C8SC05372C#!divAbstract">A graph-based genetic algorithm and generative model/Monte Carlo tree search for the exploration of chemical space. </a>          |-|<a href="https://github.com/jensengroup/GB-GA">Yes </a>|

### Zhavoronkov et al
#### [Deep learning enables the rapid identification of potent DDR1 kinase inhibitors](https://www.nature.com/articles/s41587-019-0224-x)
> This paper sparked much public press. Uses deep RL to optimize for properties in an GRU-encoded latent space.

### Jensen
#### [A graph-based genetic algorithm and generative model/Monte Carlo tree search for the exploration of chemical space. ](https://pubs.rsc.org/en/content/articlelanding/2019/SC/C8SC05372C#!divAbstract)
> Genetic algorithm that performs well on GuacaMol benchmarks.


## Predictive

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Wang  |`Huang`      | [SMILES-BERT: Large Scale Unsupervised Pre-Training for Molecular Property Prediction](https://dl.acm.org/citation.cfm?id=3342186)  |   -   | -|
|Cortes-Ciriano  |`Bender`      | <a href="https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0364-5">KekuleScope: prediction of cancer cell line sensitivity and compound potency using convolutional neural networks trained on compound images</a>    |   2   | <a href="https://github.com/isidroc/kekulescope/blob/master/Kekulescope.py">Yes</a> |
|Lee  |`Lee`      | <a href="https://www.pnas.org/content/116/9/3373">Ligand biological activity predicted by cleaning positive and negative chemical correlations</a>    |    2  | <a href="https://github.com/alphaleegroup/RandomMatrixDiscriminant">Yes</a> |
|Withnall  |`Chen`      | [Attention and Edge Memory Convolution for Bioactivity Prediction](https://link.springer.com/chapter/10.1007/978-3-030-30493-5_69) |    -  | - |

  

### Withnall, Lindelöf, Engkvist, Chen
#### [Attention and Edge Memory Convolution for Bioactivity Prediction](https://link.springer.com/chapter/10.1007/978-3-030-30493-5_69)
> One of the first few examples of neural attention being used in drug discovery.

### Wang, Guo, Wang, Sun, Huang
#### [SMILES-BERT: Large Scale Unsupervised Pre-Training for Molecular Property Prediction](https://dl.acm.org/citation.cfm?id=3342186)
> Taking inspiration from the recent monumental progress in NLP, Wang applies Google's language-model ideas to massive amounts of chemical data.


### Cortes-Ciriano, Bender
#### [ KekuleScope: prediction of cancer cell line sensitivity and compound potency using convolutional neural networks trained on compound images](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0364-5)
> Using modern CNN architectures and transfer learning from ImageNet to predict activity from RDKit-rendered skeletal structures of the ligand.

### Lee, Yang, Bassyouni, Butler, Hou, Jenkinson, Price
#### [Ligand biological activity predicted by cleaning positive and negative chemical correlations](https://www.pnas.org/content/116/9/3373)
> Lee's original RMT (random matrix theory) algorithm is extended to incorporate information from inactive compounds.


## Reaction Prediction and Retrosynthesis

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-----------------------------|-----------------|---|
|Schwaller  |`Lee`      |<a href="https://pubs.acs.org/doi/abs/10.1021/acscentsci.9b00576">Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction</a>       | 6 |<a href="https://github.com/pschwllr/MolecularTransformer">Yes</a>
|Lee  |`Lee`      |<a href="https://pubs.rsc.org/en/content/articlelanding/2019/cc/c9cc05122h#!divAbstract">Molecular Transformer unifies reaction prediction and retrosynthesis across pharma chemical space</a>       | - | -

### Schwaller, Laino, Gaudin, Bolgar, Hunter, Bekas, Lee
#### <a href="https://pubs.acs.org/doi/abs/10.1021/acscentsci.9b00576">Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction</a> 
>Uses the latest cutting-edge research from the NLP community (transformer networks), viewing reaction prediction as a machine translation problem. Word1/Molecule1 + Word2/Molecule2 ----translates_to----> Word3/Molecule3


### Lee, Yang, Sresht, Bolgar, Hou, Klug-McLeod, Butler
#### <a href="https://pubs.rsc.org/en/content/articlelanding/2019/cc/c9cc05122h#!divAbstract">Molecular Transformer unifies reaction prediction and retrosynthesis across pharma chemical space</a> 
>Applies the above technology to retrosynthesis



# 2018



## General
|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-----------------------------|-----------------|---|
|Wu  |`Pande`      |<a href="https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a#!divAbstract">MoleculeNet: a benchmark for molecular machine learning</a>       | 208 |<a href="https://github.com/deepchem/deepchem">DeepChem</a>
| Chen|`Blaschke`      |<a href="https://www.sciencedirect.com/science/article/pii/S1359644617303598">The rise of deep learning in drug discovery</a>  | 169 |-
| O'Boyle|`NextMove`      |[DeepSMILES: An Adaptation of SMILES for Use in Machine-Learning of Chemical Structures](https://chemrxiv.org/articles/DeepSMILES_An_Adaptation_of_SMILES_for_Use_in_Machine-Learning_of_Chemical_Structures/7097960/1)| 5 | [Yes](https://github.com/nextmovesoftware/deepsmiles)

### Chen, Engkvist, Wang, Olivecrona, Blaschke
#### <a href="https://www.sciencedirect.com/science/article/pii/S1359644617303598">The rise of deep learning in drug discovery</a> 
> A review on the latest developments in the field.

### Wu, Ramsundar, Feinberg, Gomes, Geniesse, Pappu, Leswing, Pande
#### <a href="https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a#!divAbstract">MoleculeNet: a benchmark for molecular machine learning</a> 
>The Pande group curated a set of datasets to assess the quality of a machine learning model on chemistry/drug discovery/molecular problems.


## Generative

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-----------------------------|-----------------|---|
|Segler|`Benevolent`     |<a href="https://pubs.acs.org/doi/10.1021/acscentsci.7b00512">Generating Focused Molecule Libraries for Drug Discovery with Recurrent Neural Networks</a>           |188|<a href="https://github.com/insilicomedicine/GENTRL">Yes</a>|
|Jin     |`Jaakkola`            |<a href="https://pubs.rsc.org/en/content/articlelanding/2019/SC/C8SC05372C#!divAbstract">Junction Tree Variational Autoencoder for Molecular Graph Generation. </a>          |93|<a href="https://github.com/wengong-jin/icml18-jtnn">Yes </a>|
|Popova     |`Tropsha`            |<a href="https://advances.sciencemag.org/content/4/7/eaap7885">Deep reinforcement learning for de novo drug design </a>          |82|<a href="https://github.com/isayev/ReLeaSE">Yes </a>|


### Segler, Kogej, Tyrchan, Waller
#### [Generating Focused Molecule Libraries for Drug Discovery with Recurrent Neural Networks](https://pubs.acs.org/doi/10.1021/acscentsci.7b00512)
> A straightforward example of the application of RNNs (specifically LSTMs) to generation of molecules and exploration of chemical space, using SMILES as input featurization.

### Jin, Barzilay, Jaakkola
#### <a href="https://arxiv.org/abs/1802.04364">Junction Tree Variational Autoencoder for Molecular Graph Generation</a> 
> Featurizes molecules based on its component fragments. Large improvement over other featurizations in many tasks.

### Popova, Isayev, Tropsha
#### [Deep reinforcement learning for de novo drug design](https://advances.sciencemag.org/content/4/7/eaap7885)
> Illustration of the use of deep reinforcement learning to generate molecules with a bias towards certain properties (bioactivity, logP etc). Features the use of a Stack Neural Network for encoding, introduced by Facebook researchers recently.



## Retrosynthesis and Reaction Prediction

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Segler  |`Waller`      | [Planning chemical syntheses with deep neural networks and symbolic AI](https://www.nature.com/articles/nature25978)|   216 | -|

### Segler, Waller
#### [Planning chemical syntheses with deep neural networks and symbolic AI](https://www.nature.com/articles/nature25978)
>Nature paper on automated retrosynthesis. Clever data augmentation methods to generate more negative data.



# 2017

## General
|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Goh  |`Vishnu`      | <a href="https://arxiv.org/abs/1704.01212">Deep Learning for Computational Chemistry</a>    |   157 | - |
|Wallach  |`Atomwise`      |[Most Ligand-Based Classification Benchmarks Reward Memorization Rather than Generalization](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00403)    |   20 | - |
|Axen  |`Keiser`      | <a href="https://pubs.acs.org/doi/10.1021/acs.jmedchem.7b00696">A Simple Representation of Three-Dimensional Molecular Structure</a>    |   9 |  <a href="https://github.com/keiserlab/e3fp">Yes</a> |



### Goh, Hodas, Vishnu
#### <a href="https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.24764">Deep Learning for Computational Chemistry</a>
> A review of the current use cases for deep learning in computational chemistry.




### Wallach, Heifets
#### [Most Ligand-Based Classification Benchmarks Reward Memorization Rather than Generalization](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00403)
>Introduction of the Asymmetric Validation Embedding (AVE) bias to better assess the domain of applicability of a machine learning model.




### Axen, Huang, Caceres, Gendelev, Roth, Keiser
#### <a href="https://pubs.acs.org/doi/abs/10.1021/acs.jmedchem.7b00696">A Simple Representation of Three-Dimensional Molecular Structure</a>
> First introduction of the concept of a 3D fingerprint. Performs moderately in benchmarks.

## Predictive

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Gilmer  |`Google`      | <a href="https://arxiv.org/abs/1704.01212">Neural Message Passing for Quantum Chemistry</a>    |   522 | - |
|Faber  |`Lilienfeld/Google`      | [Prediction Errors of Molecular Machine Learning Models Lower than Hybrid DFT Error](https://pubs.acs.org/doi/full/10.1021/acs.jctc.7b00577 "Prediction Errors of Molecular Machine Learning Models Lower than Hybrid DFT Error")    |   143 | - |
|Goh  |`Vishnu`      | <a href="https://arxiv.org/abs/1706.06689">Chemception: A Deep Neural Network with Minimal Chemistry Knowledge Matches the Performance of Expert-developed QSAR/QSPR Models</a>    |   44 | - |
|Kearnes  |`Vertex/Pande`      | <a href="https://arxiv.org/pdf/1606.08793.pdf">Modeling Industrial ADMET Data with Multitask Networks</a>    |   30 | - |




### Gilmer, Schoenholz, Riley, Vinyals, Dahl
#### <a href="https://arxiv.org/abs/1704.01212">Neural Message Passing for Quantum Chemistry</a>
> Message passing neural networks on molecular graphs are shown to outperform previous state-of-the-art on a quantum chemistry dataset (QM9)


### Goh, Siegel, Vishnu, Hodas, Baker
#### <a href="https://arxiv.org/abs/1706.06689">Chemception: A Deep Neural Network with Minimal Chemistry Knowledge Matches the Performance of Expert-developed QSAR/QSPR Models</a>
> One of the first examples of (almost naively) applying convolutional neural networks to pictures of molecules. Surprisingly (or not) it performs as well as conventional models that require domain knowledge to create.

### Kearnes, Goldman, Pande
#### <a href="https://arxiv.org/pdf/1606.08793.pdf">Modeling Industrial ADMET Data with Multitask Networks</a>
> A solid use case of multitask networks.

### Faber, Hutchison, Huang, Gilmer, Schoenholz, Dahl, Vinyals, Kearnes, Riley, Lilienfeld
#### [Prediction Errors of Molecular Machine Learning Models Lower than Hybrid DFT Error](https://pubs.acs.org/doi/full/10.1021/acs.jctc.7b00577 "Prediction Errors of Molecular Machine Learning Models Lower than Hybrid DFT Error")
> A thorough of the power of machine learning models, and in particular, deep methods, to the application of prediction of quantum mechanical properties of molecules. Suggests that given enough data with electron correlation, ML models could outperform hybrid DFT.



## Generative

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Sanchez-Lengeling  |`Aspuru-Guzik`      | <a href="https://chemrxiv.org/articles/ORGANIC_1_pdf/5309668">Optimizing distributions over molecular space. An Objective-Reinforced Generative Adversarial Network for Inverse-design Chemistry (ORGANIC)</a>    |   46 | [Yes](https://github.com/aspuru-guzik-group/ORGANIC) |

### Sanchez-Lengeling, Outeiral, Guimaraes, Aspuru-Guzik

[Optimizing distributions over molecular space. An Objective-Reinforced Generative Adversarial Network for Inverse-design Chemistry (ORGANIC)](https://chemrxiv.org/articles/ORGANIC_1_pdf/5309668)
> Uses the recently popular combination of combining a GAN with reinforcement learning to direct generative examples towards a defined prior.


## Retrosynthesis and Reaction Prediction

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Segler  |`Waller`      | [Neural-Symbolic Machine Learning for Retrosynthesis and Reaction Prediction](https://onlinelibrary.wiley.com/doi/abs/10.1002/chem.201605499)|   94 | -|
|Liu  |`Pande`      | [Retrosynthetic Reaction Prediction Using Neural Sequence-to-Sequence Models](https://pubs.acs.org/doi/10.1021/acscentsci.7b00303)|   70 | -|

### Segler, Waller
#### [Neural-Symbolic Machine Learning for Retrosynthesis and Reaction Prediction](https://onlinelibrary.wiley.com/doi/abs/10.1002/chem.201605499)
>One of the first examples of the use of RNNs for reaction prediction and retrosynthesis. Makes use of the attention mechanism.


# 2016


## General


## Predictive

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Kearnes  |`Google/Pande`      | <a href="https://link.springer.com/article/10.1007/s10822-016-9938-8">Molecular Graph Convolutions: Moving Beyond Fingerprints</a>    |   327   |-|
|Lee  |`Brenner/Colwell`      | <a href="https://www.pnas.org/content/113/48/13564">Predicting protein-ligand affinity with a random matrix framework</a>    |   20   | <a href="https://github.com/aspuru-guzik-group/chemical_vae">Yes</a> |



### Kearnes, McCloskey, Berndl, Pande, Riley
#### <a href="https://link.springer.com/article/10.1007/s10822-016-9938-8">Molecular Graph Convolutions: Moving Beyond Fingerprints</a>
> Further demonstrate of the possible merits in using graph convolutions for molecular machine learning.

### Lee, Brenner, Colwell

#### <a href="https://www.pnas.org/content/113/48/13564.short">Predicting protein-ligand affinity with a random matrix framework</a>
> Development of a simple algorithm based on PCA (principle component analysis) and RMT (random matrix theory) to classify bioactivity of molecules, and gain interpretability of pharmacophores.





# 2015

## Generative
|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Bombarelli  |`Aspuru-Guzik`      | <a href="https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572#targetText=Continuous%20representations%20of%20molecules%20allow,structures%2C%20or%20interpolating%20between%20molecules.">Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules</a>    |   410   | <a href="https://github.com/aspuru-guzik-group/chemical_vae">Yes</a> |

### Bombarelli, Wei, Duvenaud, Hernandez-Lobato, Sanchez-Lengeling, Sheberla, Aguilera-Iparraguirre, Hirzel, Adams, Aspuru-Guzik

#### [https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572](https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572)

> A SMILES variational autoencoder maps molecules to a latent space, which is continuous and differentiable, and can be optimized for certain properties (logP, QED, SAS, bioactivity etc)

## Predictive

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Duvenaud  |`Aspuru-Guzik/Adams`      | <a href="https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572#targetText=Continuous%20representations%20of%20molecules%20allow,structures%2C%20or%20interpolating%20between%20molecules.">Convolutional Networks on Graphs for Learning Molecular Fingerprints</a>    |   749   | - |
|Ma  |`Sheridan`      | <a href="https://pubs.acs.org/doi/10.1021/ci500747n">Deep Neural Nets as a Method for Quantitatitve Structure-Activity Relationships</a>    |   380   | - |
|Ramsundar  |`Google/Pande`      | <a href="https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572#targetText=Continuous%20representations%20of%20molecules%20allow,structures%2C%20or%20interpolating%20between%20molecules.">Massively Multitask Networks for Drug Discovery</a>    |   222   | <a href="https://github.com/deepchem/deepchem">DeepChem</a> |
|Wallach  |`Atomwise`      | <a href="https://arxiv.org/abs/1510.02855">AtomNet: A Deep Convolutional Neural Network for Bioactivity Prediction in Structure-based Drug Discovery</a>    |   165   | - |



### Duvenaud, Maclaurin, Aguilera-Iparraguirre, Gomez-Bombarelli, Aspuru-Guzik, Adams

#### [Convolutional Networks on Graphs for Learning Molecular Fingerprints](http://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints)

>First example of elucidating the potential of graph convolutions on molecules.

### Ma, Sheridan, Liaw, Dahl, Svetnik

#### [Deep Neural Nets as a Method for Quantitatitve Structure-Activity Relationships](https://pubs.acs.org/doi/10.1021/ci500747n)

> Follow-up paper to the Merck Kaggle challenge, which was won by a researcher in Hinton's lab. One of the first examples of the pushing of deep learning into the limelight for drug discovery.



### Ramsundar, Kearnes, Riley
#### [Massively Multitask Networks for Drug Discovery](https://arxiv.org/abs/1502.02072)

> Using a shared representation of hundreds of thousands of molecules to predict activity at multiple targets simultaneously. Some analysis is done to elucidate on the multitask effect.


### Wallach, Dzamba, Heifets

#### [AtomNet: A Deep Convolutional Neural Network for Bioactivity Prediction in Structure-based Drug Discovery](https://arxiv.org/abs/1510.02855)

> First known example of CNNs being applied to ligand-based drug discovery in the literature.



# 2014

## Predictive

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Dahl  |`Salakhutdinov`      | <a href="https://arxiv.org/pdf/1406.1231.pdf">Multi-task Neural Networks for QSAR Predictions</a>    |   156   |-|


### Dahl, Jaitly, Salakhutdinov
#### [Multi-task Neural Networks for QSAR Predictions](https://arxiv.org/pdf/1406.1231.pdf)
> First description of multi-task networks for drug discovery in the literature. Provides a short account of their application in the Merck Kaggle challenge of 2012.



# 2013

## General 

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Sheridan  |`Sheridan`      | <a href="https://pubs.acs.org/doi/full/10.1021/ci400084k">Time-Split Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction</a>    |   84   |-|


### Sheridan
#### [Time-Split Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction](https://pubs.acs.org/doi/full/10.1021/ci400084k)
> Sheridan argues that random splitting of train and test sets results in too optimistic predictions, whereas scaffold-based splitting is too pessimistic. Time-validation splits are the most realistic split and corresponds to the data a model will face when deployed.




# 2012

## General

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Bickerton  |`Exscientia`      | <a href="https://www.nature.com/articles/nchem.1243">Quantifying the chemical beauty of drugs</a>    |   420   | <a href="https://www.rdkit.org/docs/source/rdkit.Chem.QED.html">Yes</a> |



### Bickerton, Paolini, Besnard, Muresan, Hopkins
#### [Quantifying the chemical beauty of drugs](https://www.nature.com/articles/nchem.1243)
> Introduction of a metric to assess general drug-likeness based on modelling probability distributions for Lipinski's 5 paramters using a curated set of orally active pharmaceuticals.

## Generative

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Besnard    |`Exscientia`      | <a href="https://www.nature.com/articles/nature11691">Automated design of ligands to polypharmacological profiles</a>   |   454   | - |


### Besnard et al
#### [Automated design of ligands to polypharmacological profiles](https://www.nature.com/articles/nature11691)
>Multi-objective optimization using Bayesian methods. Generative design using priors derived from medicinal chemistry.




## Predictive

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Montavon  |`Lilienfeld`      | <a href="https://papers.nips.cc/paper/4830-learning-invariant-representations-of-molecules-for-atomization-energy-prediction">Learning Invariant Representations of Molecules for Atomization Energy Prediction</a>    |   85   | <a href="http://quantum-machine.org/datasets/">Yes</a> |
|Chen  |`Voigt`      | <a href="https://pubs.acs.org/doi/10.1021/ci200615h">Comparison of Random Forest and Pipeline Pilot Naive Bayes in Prospective QSAR Predictions</a>    |   60   | <a href="https://github.com/aspuru-guzik-group/chemical_vae">Yes</a> |


### Montavon, Hansen, Fazil, Rupp, Biegler, Ziehe, Tkatchenko, Lilienfeld, Muller
#### <a href="https://papers.nips.cc/paper/4830-learning-invariant-representations-of-molecules-for-atomization-energy-prediction">Learning Invariant Representations of Molecules for Atomization Energy Prediction</a>
> First example of the use of the Coulomb matrix for inferring quantum mechanical properties of the molecule.

### Chen, Sheridan, Hornak, Voigt
#### [Comparison of Random Forest and Pipeline Pilot Naïve Bayes in Prospective QSAR Predictions](https://pubs.acs.org/doi/10.1021/ci200615h)
> These authors argue that, although Random Forest is computationally expensive, it often outperforms Naive Bayes significantly.


# 2010


## General

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Rogers  |`?`      | <a href="https://pubs.acs.org/doi/10.1021/ci100050t">Extended-connectivity fingerprints</a>    |   1638   | - |


### Rogers, Hahn
#### [Extended-connectivity fingerprints](https://pubs.acs.org/doi/10.1021/ci100050t)
>Fingerprints are one of the best featurizations for chemoinformatics and machine-learning tasks for chemistry. A key paper in the field.


# 1998

## General

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Weininger  |`Weininger`      | [SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules](https://pubs.acs.org/doi/10.1021/ci00057a005)   |   2719   | - |
