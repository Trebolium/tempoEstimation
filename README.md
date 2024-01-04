# tempoEstimationTask

## Task Description
Consider entire song before detecting beat
Not a Classification problem
Imbalance between beats and no beats (dixon2001, yan2023)
-   Weighted losses, smoothed labels    
There is a Long sequential inputs issue (yan2023)
10ms frames are commonly used, but lead to imbalance, and long sequences

## Considerations
Task is not beat detection. Therefore no beat labels necessary
Choose only realistic values for tempo (30 - 285) (sun2021)
How lightweight do we want the model to be:
-   preprocessing
-   NNs have output probabiilities for each input frame - HMM for beat time detection as post-processing step
-   Octave errors: How important are they? Accuracy1 penalises ooctave errors, Accuracy 2 doesn't (sun2021)
Augmentation of time-strectching might be useful

### Features to use
-   Onsets - tell us where most energy accumulates at beats
    -   Inter-onset intervals
-   HPSS
    -   Librosa.effects.hpss
-   Embeddings, contrastive learning, VGGish?
-   Chromogram - provide chords for musical context
-   Spectrogram - Contains all features
-   Waveform - contains all unformation. None lost

### DSP Model
-   Pros
    -   Lightweight
    -   No iterative training necessary
    -   Makes use of expert knowledge in the field of tempo estimation
    -   Interpretable
-   Cons
    -   Good performances, but not as powerful as NN model


#### DSP Model walkthrough
    -   Load dataset from disk
    -   Convert to spectrograms
    -   Derive onset arrays from spectrograms
    -   Clean up onset arrays
    -   Perform autocorrelation
    -   Select from best autocor solutions
    -   Compare against averaged song tempo

#### Potential Pitfalls
    -   Parameters vary significatly between songs
    -   Artefacts may arise on edges of audio clips
    -   Silent/implied beats may misguide algorithm

## NN-model literature

Good resource: https://tempobeatdownbeat.github.io/tutorial/intro.html

### hung2022
-   SpecTNT for modelling beats and downbeats
-   https://github.com/MWM-io/SpecTNT-pytorch

### cheng2023a
-   Beat tracking - frame-level, sigmoid output
-   Transformer
-   Input: melspec windows (T, 80)
-   Low-res encoder, high-res decoder
-   F1 score: 88

### sun2021
-   Tempo estimation -classification problem (30-285bpm)
-   Input: melspec of 6s
-   MGANet
    -   Uses multilayer CNN of diff resolutions to capture high level representations after just a few layers
-   Accuracy1 79 (boch 69), Accuracy2 91 (boch 95)

### quinton2022
-   Tempo estimation - produce scalar embedding
-   Contrastive learning equivariance
    -   Use augmentation to time stretch examples and pair them negatively
-   Use Temporal Comvolutional Network (boch2019)
    -   Competitive performance, few parameters (33k)
-   0.70 Accuracy1, 0.95 Accuracy2
-   Input is 13.6 seconds of audio
-   https://github.com/Quint-e/equivariant-self-supervision-tempo

### heydari2021
-   CRNN
-   beat detection - 3 way classification: beat, non-beat, downbeat (softmax summed)
-   Use CQT, 24bpo, and their first order derivatives, concatenated (total 272-d)
-   329-d hand-crafted feature set from [15], which comprises chroma features, onset strengths, low-frequency spectral features, and melodic constant-Q spectral features. These did NOT improve results
-   F1 score 81 (bock was 79).

## Based on these findings
    Assess gain against risk
    Feasibilty of getting up and running
    Modularity
    Seems like beat-tracking would not go to waste, given Hooks product
        Would act as a stepping stone towards tempo estimation (which might be the easier bit)
    Quinton2022
        results sometimes better than Bock
        Coode available intigating head start
        Reduces difficulty-gain trade-off
        Modular Training
    boch2019
        https://github.com/CPJKU/madmom

