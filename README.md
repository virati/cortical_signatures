# Cortical Signatures of SCCwm-DBS

![DBS of the SCCwm targets specific tracts in the brain that likely mediate the immediate effects.](https://med.emory.edu/education/vme/assets/images/gallery/dbs-depression/dbs-for-major-depression.jpg)
<p align="center">DBS of the SCCwm targets specific tracts in the brain that likely mediate the immediate effects. Source: https://med.emory.edu/education/vme/pages/gallery/pf-dbs-depression-image.html</p>

## Overview
Deep brain stimulation (DBS) targeted at a patient's specific subcallosal cingulate white matter (SCCwm) alleviates symptoms of depression.
But how this SCCwm-DBS affects the brain, where and how does it change ongoing brain dynamics, remains unclear.

In this repository I address the question: what does SCCwm-DBS do immediately to brain dynamics across multiple scales of measurement.

## Requirements
This repository requires the custom library ```DBSpace``` [link](https://github.com/virati/DBSpace)

## Network Action
SCCwm-DBS is assumed to effect changes in brain dynamics - we want to know _where_ and _how_ these changes manifest.
Together, this makes up the _network action_ of SCCwm-DBS and, using a combination of intracranial local field potentials (LFPs) and scalp dense-array electroencephalography (dEEG), I characterize the network action at therapeutic stimulation parameters.

Characterization of the network action is done in two parts: spatial modes and temporal modes.

### Spatial Modes
[In Prep]

### Temporal Modes
With an eye towards temporal changes, we studied the immediate effects of SCCwm-DBS on the trajectory of whole brain oscillatory activity.

Publication: [here](https://www.frontiersin.org/articles/10.3389/fnins.2022.768355/full)
The code for this publication is available in the ```analysis/DOs``` folder.


## Target Engagement Classifier
If we can confirm adequate stimulation of the SCCwm using neural recordings, we may not need to wake up patients in the operating room, making the surgery quicker and more standardized.
Using the dEEG I developed a classifier that can confirm adequate SCCwm-DBS.

### Offline Classifier
[In Prep]

### Online Classifier
[In Prep]
