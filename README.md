#  AI in Drug Discovery Progress

This repository contains an up-to-date list (as of September 2019) of progress (papers, github repos etc) made in applying AI to drug discovery. 


# 2019

## General
|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-----------------------------|-----------------|---|
|Brown  |`Benevolent`      |<a href="https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839">GuacaMol: Benchmarking Models for de Novo Molecular Design</a>       | 13 |<a href="https://github.com/BenevolentAI/guacamol">Yes</a>

### Brown, Fiscato, Segler, Vaucher
#### <a href="https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839">GuacaMol: Benchmarking Models for de Novo Molecular Design</a> 
>BenevolentAI designed a set of benchmarks used to assess the quality of generative models.





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
> Seems to do well on Guacamol benchmarks.


## Predictive

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Cortes-Ciriano  |`Bender`      | <a href="https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0364-5">KekuleScope: prediction of cancer cell line sensitivity and compound potency using convolutional neural networks trained on compound images</a>    |   2   | <a href="https://github.com/isidroc/kekulescope/blob/master/Kekulescope.py">Yes</a> |
|Lee  |`Lee`      | <a href="https://www.pnas.org/content/116/9/3373">Ligand biological activity predicted by cleaning positive and negative chemical correlations</a>    |    2  | <a href="https://github.com/alphaleegroup/RandomMatrixDiscriminant">Yes</a> |
### Cortes-Ciriano, Bender
#### [ KekuleScope: prediction of cancer cell line sensitivity and compound potency using convolutional neural networks trained on compound images](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0364-5)
> Using modern CNN architectures and transfer learning from ImageNet to predict activity from RDKit-rendered skeletal structures of the ligand.

### Lee, Yang, Bassyouni, Butler, Hou, Jenkinson, Price
#### [Ligand biological activity predicted by cleaning positive and negative chemical correlations](https://www.pnas.org/content/116/9/3373)
> Lee's original RMT (random matrix theory) algorithm is extended to incorporate information from inactive compounds.


# 2018



## General
|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-----------------------------|-----------------|---|
|Wu  |`Pande`      |<a href="https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a#!divAbstract">MoleculeNet: a benchmark for molecular machine learning</a>       | 208 |<a href="https://github.com/deepchem/deepchem">DeepChem</a>

### Wu, Ramsundar, Feinberg, Gomes, Geniesse, Pappu, Leswing, Pande
#### <a href="https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a#!divAbstract">MoleculeNet: a benchmark for molecular machine learning</a> 
>The Pande group curated a set of datasets to assess the quality of a machine learning model on chemistry/drug discovery/molecular problems.


## Generative

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-----------------------------|-----------------|---|
|Segler|`Benevolent`     |<a href="https://pubs.acs.org/doi/10.1021/acscentsci.7b00512">Generating Focused Molecule Libraries for Drug Discovery with Recurrent Neural Networks</a>           |188|<a href="https://github.com/insilicomedicine/GENTRL">Yes</a>|
|Jin     |`Jaakkola`            |<a href="https://pubs.rsc.org/en/content/articlelanding/2019/SC/C8SC05372C#!divAbstract">Junction Tree Variational Autoencoder for Molecular Graph Generation. </a>          |93|<a href="https://github.com/wengong-jin/icml18-jtnn">Yes </a>|


### Segler, Kogej, Tyrchan, Waller
#### [Generating Focused Molecule Libraries for Drug Discovery with Recurrent Neural Networks](https://pubs.acs.org/doi/10.1021/acscentsci.7b00512)
> A straightforward example of the application of RNNs (specifically LSTMs) to generation of molecules and exploration of chemical space, using SMILES as input featurization.

### Jin, Barzilay, Jaakkola
#### <a href="https://arxiv.org/abs/1802.04364">Junction Tree Variational Autoencoder for Molecular Graph Generation</a> 
> Featurizes molecules based on its component fragments. Large improvement over other featurizations in many tasks.


# 2017




# 2016

## Predictive

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Lee  |`Brenner/Colwell`      | <a href="https://link.springer.com/article/10.1007/s10822-016-9938-8">Predicting protein-ligand affinity with a random matrix framework</a>    |   20   | <a href="https://github.com/aspuru-guzik-group/chemical_vae">Yes</a> |




### Lee, Brenner, Colwell

#### <a href="https://www.pnas.org/content/113/48/13564.short">Predicting protein-ligand affinity with a random matrix framework</a>
> Lee, Brenner and Colwell develop a simple algorithm based on PCA (principle component analysis) and RMT (random matrix theory) to classify bioactivity of molecules, and gain interpretability of pharmacophores.

## Review
### Cortes-Ciriano, Mervin, Bender
#### [Current Trends in Drug Sensitivity Prediction](http://www.eurekaselect.com/146734/article)


|






# 2015

## Generative
|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Bombarelli  |`Aspuru-Guzik`      | <a href="https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572#targetText=Continuous%20representations%20of%20molecules%20allow,structures%2C%20or%20interpolating%20between%20molecules.">Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules</a>    |   410   | <a href="https://github.com/aspuru-guzik-group/chemical_vae">Yes</a> |

### Bombarelli, Wei, Duvenaud, Hernandez-Lobato, Sanchez-Lengeling, Sheberla, Aguilera-Iparraguirre, Hirzel, Adams, Aspuru-Guzik

#### [https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572](https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572)

> Aspuru-Guzik's group design a SMILES variational autoencoder to map molecules to a latent space. A molecule's latent representation, which is now continuous and differentiable, can be optimized for certain properties (logP, QED, SAS, bioactivity etc)

## Predictive

|Lead Author|Group|Title|Citations|Code|
|-----------|---------|-------------------------------|-----------------|---|
|Ma  |`Sheridan`      | <a href="https://pubs.acs.org/doi/10.1021/ci500747n">Deep Neural Nets as a Method for Quantitatitve Structure-Activity Relationships</a>    |   380   | <a href="https://github.com/aspuru-guzik-group/chemical_vae">Yes</a> |
|Wallach  |`Atomwise`      | <a href="https://arxiv.org/abs/1510.02855">AtomNet: A Deep Convolutional Neural Network for Bioactivity Prediction in Structure-based Drug Discovery</a>    |   165   | <a href="https://github.com/aspuru-guzik-group/chemical_vae">Yes</a> |
|Duvenaud  |`Aspuru-Guzik/Adams`      | <a href="https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572#targetText=Continuous%20representations%20of%20molecules%20allow,structures%2C%20or%20interpolating%20between%20molecules.">Convolutional Networks on Graphs for Learning Molecular Fingerprints</a>    |   749   | <a href="https://github.com/aspuru-guzik-group/chemical_vae">Yes</a> |
|Ramsundar  |`Google/Pande`      | <a href="https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572#targetText=Continuous%20representations%20of%20molecules%20allow,structures%2C%20or%20interpolating%20between%20molecules.">Massively Multitask Networks for Drug Discovery</a>    |   222   | <a href="https://github.com/deepchem/deepchem">DeepChem</a> |
### Ma, Sheridan, Liaw, Dahl, Svetnik

#### [Deep Neural Nets as a Method for Quantitatitve Structure-Activity Relationships](https://pubs.acs.org/doi/10.1021/ci500747n)

> Follow-up paper to the Merck Kaggle challenge, which was won by a researcher in Hinton's lab. One of the first examples of the pushing of deep learning into the limelight for drug discovery.


### Wallach, Dzamba, Heifets

#### [AtomNet: A Deep Convolutional Neural Network for Bioactivity Prediction in Structure-based Drug Discovery](https://arxiv.org/abs/1510.02855)

> First known example of CNNs being applied to ligand-based drug discovery in the literature.

### Duvenaud, Maclaurin, Aguilera-Iparraguirre, Gomez-Bombarelli, Aspuru-Guzik, Adams

#### [Convolutional Networks on Graphs for Learning Molecular Fingerprints](http://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints)

>First example of elucidating the potential of graph convolutions on molecules.


### Ramsundar, Kearnes, Riley
#### [Massively Multitask Networks for Drug Discovery](https://arxiv.org/abs/1502.02072)

> Using a shared representation of hundreds of thousands of molecules to predict activity at multiple targets simultaneously.



# 2014





# 2013






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

# 2011 

# 2010


## Predictive

### Rogers, Hahn
#### [Extended-connectivity fingerprints](https://pubs.acs.org/doi/10.1021/ci100050t)
>Fingerprints are currently the state-of-the-art (SOTA) for many chemoinformatics and machine-learning tasks for chemistry. Everyone in the field owes this paper some time.
