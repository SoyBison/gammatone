---
title: "Accelerated Gammatones"
date: 2020-05-12T14:37:01-05:00
draft: false
---

Last winter, I worked on a personal project I call [_ongaku_](https://www.coeneedell.com/projects/ongaku/) (from the Japanese for 'music'). This was an attempt to use manifold learning to create a metric space for music. The preprocessing relied heavily on a method called [_Gammatone Cepstrum Analysis_](https://ieeexplore.ieee.org/document/6202347). This method was intended to replace Mel Frequency Cepstral Coefficients. Where Mel Frequency is a logarithmic transformation of sound frequency, in an attempt to simulate human perception of sound. The most common transformation for Mels is:

$$
m = 2595 \log_{10}{1 + \frac{f}{700}}
$$

This simple approach works very well, but it has its basis in human _perception_ which is a tricky thing. There's no reason to believe that the brain makes judgments based on human perception. It's very possible that the brain takes in data, and produces two perceptions, one of the sound itself, and the other as the semiotic interpretation of that sound. Enter the Gammatone Cepstral Coefficients. This is a simplification of the process that sound signals undergo when being transferred through the cochlear nerve, the nerve that transfers data between the ear and the brain. So the theory goes, this will allow a machine learning algorithm to work with data that better simulates how the brain receives sound signals, rather than how the mind perceives them. Research has shown that GFCCs outperform MFCCs in machine learning tasks. [(Liu 2018)](https://arxiv.org/abs/1806.09010) The gammatone transformation looks like:

$$
g_q(t) = t^4 e^{-54 \pi t + j 20 \pi t}u(t)
$$

I have been able to find an [implementation](https://github.com/detly/gammatone) of the gammatone transformation in python, but it's slow, and is a port of a MATLAB plugin. This package is intended to mimic that package's structure, but instead working natively in OpenCL for speed. This is problematic for a number of reasons. First, the MATLAB plugin has been shown to be inefficiently implemented. [(Ma n.d.)](https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources/gammatone/). Secondly, for my purposes, gammatones need to be processed on long files, 2-10 minutes in length, and even Dr. Ma's implementation runs serially. As such, I believe that an inherently parallel version of this would be a benefit, not only to me, but to the scientific computing community at large. As such, this project will be released on pypi as `gammatone`. There does exist a github project called `gammatone` which is the implementation that ongaku currently uses, but it is not avaliable on pypi at the time of writing, so there will be no conflicts in the long run. Ideally this project will end up somewhere between Ma's implementation and detly's implementation.