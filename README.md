---
title: "Accelerated Gammatones"
author: Coen D. Needell
date: 2020-05-12T14:37:01-05:00
draft: false
math: true
---
<h1 id="introduction">Introduction</h1>
<p>Last winter, I worked on a personal project I call <a href="https://www.coeneedell.com/projects/ongaku/"><em>ongaku</em></a> (from the Japanese for ‘music’). This was an attempt to use manifold learning to create a metric space for music. The preprocessing relied heavily on a method called <span class="citation" data-cites="valeroGammatone2012">(Valero and Alias 2012)</span>. This method was intended to replace Mel Frequency Cepstral Coefficients. Where Mel Frequency is a logarithmic transformation of sound frequency, in an attempt to simulate human perception of sound. The most common transformation for Mels is:</p>
<p><span class="math display">\[
m = 2595 \log_{10}{1 + \frac{f}{700}}
\]</span></p>
<p>This simple approach works very well, but it has its basis in human <em>perception</em> which is a tricky thing. There’s no reason to believe that the brain makes judgments based on human perception. It’s very possible that the brain takes in data, and produces two perceptions, one of the sound itself, and the other as the semiotic interpretation of that sound. Enter the Gammatone Cepstral Coefficients. This is a simplification of the process that sound signals undergo when being transferred through the cochlear nerve, the nerve that transfers data between the ear and the brain. So the theory goes, this will allow a machine learning algorithm to work with data that better simulates how the brain receives sound signals, rather than how the mind perceives them. Research has shown that GFCCs outperform MFCCs in machine learning tasks <span class="citation" data-cites="liuEvaluating2018">(Liu 2018)</span>. The gammatone transformation looks like:</p>
<p><span class="math display">\[
g_q(t) = t^4 e^{-54 \pi t + j 20 \pi t}u(t)
\]</span></p>
<p>I have been able to find an <a href="https://github.com/detly/gammatone">implementation</a> of the gammatone transformation in python, but it’s slow, and is a port of a MATLAB plugin. This package is intended to mimic that package’s structure, but instead working natively in OpenCL for speed. This is problematic for a number of reasons. First, the MATLAB plugin has been shown to be inefficiently implemented <span class="citation" data-cites="maEfficient">(Ma, n.d.a)</span> <span class="citation" data-cites="maCochleagram">(Ma, n.d.b)</span>. Secondly, for my purposes, gammatones need to be processed on long files, 2-10 minutes in length, and even Dr. Ma’s implementation runs serially. As such, I believe that an inherently parallel version of this would be a benefit, not only to me, but to the scientific computing community at large. As such, this project will be released on pypi as <code>gammatone</code>. There does exist a github project called <code>gammatone</code> which is the implementation that ongaku currently uses, but it is not avaliable on pypi at the time of writing, so there will be no conflicts in the long run.</p>
<h1 id="implementation">Implementation</h1>
<p>The available implementations of gammatone filters are serial. However, there have been a number of attempts to parallelize IIR (Infinite Impulse Response) filters in the past<span class="citation" data-cites="anwarDigital">(Anwar and Sung, n.d.)</span> <span class="citation" data-cites="bellochMultichannel2014">(Belloch et al. 2014)</span>. The general consensus is that this is rarely better than a serial implementation, and the implementation of a new filter needs to be made bespoke for that filter. After analyzing the implementation methods, I decided that (for now,) this is beyond the scope of this project. The fourth order gammatone filter is implemented like:</p>
<p><span class="math display">\[
erb(x) = 24.7 * (4.37 \times 10^{-3} + 1) \]</span> <span class="math display">\[
\delta = e^{\frac{2 \pi}{f} erb(f_c) \times 1.019} \]</span> <span class="math display">\[
q_t = \cos{\frac{2 \pi}{f} f_c} + i\sin{\frac{2 \pi}{f} f_c} \]</span> <span class="math display">\[
g = \frac{\left(\frac{2 \pi}{cf} erb(f_c) \times 1.019\right)^4}{3} \]</span> <span class="math display">\[
y_{t} =  q_t x_t + 4 \delta y_{t-1} - 6 \delta^2 y_{t-2} + 4 \delta^3 y_{t-3} - \delta^4 y_{t-4}
\]</span></p>
<p>Where <span class="math inline">\(f\)</span> is the sampling frequency of the signal, and <span class="math inline">\(f_c\)</span> is the target frequency to test membrane resonance against. The basilar membrane displacement (<span class="math inline">\(B_t\)</span>) and Hilbert envelope (<span class="math inline">\(H_i\)</span>) are defined with simple transformations. The cochleagram uses the Hilbert envelope to construct an image.</p>
<p><span class="math display">\[
B_t = g y_t q_t \]</span> <span class="math display">\[
H_t = g \sqrt{y_t^2}
\]</span></p>
<p>Another option is to use a prefix sum <span class="citation" data-cites="blellochPrefix">(Blelloch, n.d.)</span>. The problem with this is that while a single prefix sum can be exact to the first order of a filter, most implementations of gammatone cochleagrams use a fourth order filter. However, most implementations are made for medical applications, and I am a social scientist. Implementing a first order gammatone filter can, unlike higher order filters, can be reduced to a one-dimensional prefix sum operation <span class="citation" data-cites="blellochPrefix">(Blelloch, n.d.)</span>. A prefix sum is generally defined with an operator <span class="math inline">\(a \oplus b\)</span> such that:</p>
<p><span class="math display">\[
y_t = \bigoplus_{\forall t} x_t \]</span> <span class="math display">\[
y_t = x_0 \oplus x_1 \oplus \cdots \oplus x_t
\]</span></p>
<p>For a first order gammatone filter we can define:</p>
<p><span class="math display">\[
a \oplus b = \alpha \delta a + b
\]</span></p>
<p>Then implementing the gammatone filter is a matter of finding the best <span class="math inline">\(\alpha\)</span>. Based on the original description of gammatone filters, we’ll set <span class="math inline">\(\alpha = -.67\)</span>. It must be negative so that the filter can resonate with the input signal <span class="citation" data-cites="shenRapid2014">(Shen, Sivakumar, and Richards 2014)</span>.</p>
<h1 id="results">Results</h1>
<p>In comparison with the detly implementation, testing across all of the songs in the album Traffic and Weather by Fountains of Wayne <span class="citation" data-cites="wayneTraffic2007">(Wayne 2007)</span>. These songs are 2 minutes, 58 seconds on average. On my laptop, which has an Intel Core i7-7700HQ CPU @ 2.8GHzx8 processor and a GeForce GTX 1080 Max-Q GPU, the detly implementation (serial) takes 7.5 seconds to create a 16-level Cochleagram (16 gammatone filtrations) for a single song on average. The GPU implementation takes 1.75 seconds for a single song on average. This is a little more than a 4x speedup.</p>
<figure>
<img src="oldcoch.png" title="Old Cochleagram" alt="Figure 1: The cochleagram created by the serial implementation" /><figcaption>Figure 1: The cochleagram created by the serial implementation</figcaption>
</figure>
<figure>
<img src="newCoch.png" title="New Cochleagram" alt="Figure 2: The cochleagram created by the GPU implementation" /><figcaption>Figure 2: The cochleagram created by the GPU implementation</figcaption>
</figure>
<p>Figures 1 and 2 show the old and new implementation’s cochleagrams. Notice that the new implementation has a “squishy” quality to it. Which is expected since it’s a much less complicated implementation. Despite the lack of clear definition, a machine learning technique should still be able to identify aural features.</p>
<h1 id="future">Future</h1>
<p>While creating a higher order implementation of the gammatone filter and therefore the cochleagram is outside the scope of this project, I believe it can be done given more time and outside research. A possible pathway is to create a “horizontal” kernel, that runs multiple fourth order gammatone filters at once. Another approach is to convert the signal to frequency space, and then apply multiple frequency space filters to the signal in a cascading manner <span class="citation" data-cites="holdsworthImplementing">(Holdsworth, Patterson, and Nimmo-Smith, n.d.)</span>. Even though this project has resulted in a small step in the right direction, it could ultimately unlock the door for more accessible social science research on audio content.</p>
<h1 id="references" class="unnumbered">References</h1>
<div id="refs" class="references">
<div id="ref-anwarDigital">
<p>Anwar, Sajid, and Wonyong Sung. n.d. “Digital Signal Processing Filtering with GPU,” 2.</p>
</div>
<div id="ref-bellochMultichannel2014">
<p>Belloch, Jose A., Balazs Bank, Lauri Savioja, Alberto Gonzalez, and Vesa Valimaki. 2014. “Multi-Channel IIR Filtering of Audio Signals Using a GPU.” In <em>2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)</em>, 6692–6. Florence, Italy: IEEE. <a href="https://doi.org/10.1109/ICASSP.2014.6854895">https://doi.org/10.1109/ICASSP.2014.6854895</a>.</p>
</div>
<div id="ref-blellochPrefix">
<p>Blelloch, Guy E. n.d. “Prefix Sums and Their Applications,” 26.</p>
</div>
<div id="ref-holdsworthImplementing">
<p>Holdsworth, John, Roy Patterson, and Ian Nimmo-Smith. n.d. “Implementing a GammaTone Filter Bank,” 5.</p>
</div>
<div id="ref-liuEvaluating2018">
<p>Liu, Gabrielle K. 2018. “Evaluating Gammatone Frequency Cepstral Coefficients with Neural Networks for Emotion Recognition from Speech.” <em>arXiv:1806.09010 [Cs, Eess]</em>, June. <a href="http://arxiv.org/abs/1806.09010">http://arxiv.org/abs/1806.09010</a>.</p>
</div>
<div id="ref-maEfficient">
<p>Ma, Ning. n.d.a. “An Efficient Implementation of Gammatone Filters.” https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources/gammatone/.</p>
</div>
<div id="ref-maCochleagram">
<p>———. n.d.b. “Cochleagram Representaion of Sound.” https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources/ratemap/.</p>
</div>
<div id="ref-shenRapid2014">
<p>Shen, Yi, Rajeswari Sivakumar, and Virginia M. Richards. 2014. “Rapid Estimation of High-Parameter Auditory-Filter Shapes.” <em>The Journal of the Acoustical Society of America</em> 136 (4): 1857–68. <a href="https://doi.org/10.1121/1.4894785">https://doi.org/10.1121/1.4894785</a>.</p>
</div>
<div id="ref-valeroGammatone2012">
<p>Valero, X., and F. Alias. 2012. “Gammatone Cepstral Coefficients: Biologically Inspired Features for Non-Speech Audio Classification.” <em>IEEE Transactions on Multimedia</em> 14 (6): 1684–9. <a href="https://doi.org/10.1109/TMM.2012.2199972">https://doi.org/10.1109/TMM.2012.2199972</a>.</p>
</div>
<div id="ref-wayneTraffic2007">
<p>Wayne, Fountains of. 2007. “Traffic and Weather.” Virgin.</p>
</div>
</div>
the scientific computing community at large. As such, this project will be released on pypi as <code>gammatone</code>. There does exist a github project called <code>gammatone</code> which is the implementation that ongaku currently uses, but it is not avaliable on pypi at the time of writing, so there will be no conflicts in the long run.</p>
<p>The available implementations of gammatone filters are serial. However, there have been a number of attempts to parallelize IIR (Infinite Impulse Response) filters in the past<span class="citation" data-cites="anwarDigital">(Anwar and Sung, n.d.)</span> <span class="citation" data-cites="bellochMultichannel2014">(Belloch et al. 2014)</span>. The general consensus is that this is rarely better than a serial implementation, and the implementation of a new filter needs to be made bespoke for that filter. After analyzing the implementation methods, I decided that (for now,) this is beyond the scope of this project. The fourth order gammatone filter is implemented like:</p>
<p><span class="math display">\[
erb(x) = 24.7 * (4.37 \times 10^{-3} + 1) \\
\delta = e^{\frac{2 \pi}{f} erb(f_c) \times 1.019} \\
q_t = \cos{\frac{2 \pi}{f} f_c} + i\sin{\frac{2 \pi}{f} f_c} \\
g = \frac{\left(\frac{2 \pi}{cf} erb(f_c) \times 1.019\right)^4}{3} \\
y_{t} =  q_t x_t + 4 \delta y_{t-1} - 6 \delta^2 y_{t-2} + 4 \delta^3 y_{t-3} - \delta^4 y_{t-4}\\
\]</span></p>
<p>Where <span class="math inline">\(f\)</span> is the sampling frequency of the signal, and <span class="math inline">\(f_c\)</span> is the target frequency to test membrane resonance against. The basilar membrane displacement (<span class="math inline">\(B_t\)</span>) and Hilbert envelope (<span class="math inline">\(H_i\)</span>) are defined with simple transformations. The cochleagram uses the Hilbert envelope to construct an image.</p>
<p><span class="math display">\[
B_t = g y_t q_t \\
H_t = g \sqrt{y_t^2}
\]</span></p>
<p>Another option is to use a prefix sum <span class="citation" data-cites="blellochPrefix">(Blelloch, n.d.)</span>. The problem with this is that while a single prefix sum can be exact to the first order of a filter, most implementations of gammatone cochleagrams use a fourth order filter. However, most implementations are made for medical applications, and I am a social scientist. Implementing a first order gammatone filter can, unlike higher order filters, can be reduced to a one-dimensional prefix sum operation <span class="citation" data-cites="blellochPrefix">(Blelloch, n.d.)</span>.</p>
<h1 id="references" class="unnumbered">References</h1>
<div id="refs" class="references">
<div id="ref-anwarDigital">
<p>Anwar, Sajid, and Wonyong Sung. n.d. “Digital Signal Processing Filtering with GPU,” 2.</p>
</div>
<div id="ref-bellochMultichannel2014">
<p>Belloch, Jose A., Balazs Bank, Lauri Savioja, Alberto Gonzalez, and Vesa Valimaki. 2014. “Multi-Channel IIR Filtering of Audio Signals Using a GPU.” In <em>2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)</em>, 6692–6. Florence, Italy: IEEE. <a href="https://doi.org/10.1109/ICASSP.2014.6854895">https://doi.org/10.1109/ICASSP.2014.6854895</a>.</p>
</div>
<div id="ref-blellochPrefix">
<p>Blelloch, Guy E. n.d. “Prefix Sums and Their Applications,” 26.</p>
</div>
<div id="ref-liuEvaluating2018">
<p>Liu, Gabrielle K. 2018. “Evaluating Gammatone Frequency Cepstral Coefficients with Neural Networks for Emotion Recognition from Speech.” <em>arXiv:1806.09010 [Cs, Eess]</em>, June. <a href="http://arxiv.org/abs/1806.09010">http://arxiv.org/abs/1806.09010</a>.</p>
</div>
<div id="ref-maEfficient">
<p>Ma, Ning. n.d.a. “An Efficient Implementation of Gammatone Filters.” https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources/gammatone/.</p>
</div>
<div id="ref-maCochleagram">
<p>———. n.d.b. “Cochleagram Representaion of Sound.” https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources/ratemap/.</p>
</div>
<div id="ref-valeroGammatone2012">
<p>Valero, X., and F. Alias. 2012. “Gammatone Cepstral Coefficients: Biologically Inspired Features for Non-Speech Audio Classification.” <em>IEEE Transactions on Multimedia</em> 14 (6): 1684–9. <a href="https://doi.org/10.1109/TMM.2012.2199972">https://doi.org/10.1109/TMM.2012.2199972</a>.</p>
</div>
</div>
</body>
</html>
