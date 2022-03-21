# Cortical Signatures of SCCwm-DBS

![DBS of the SCCwm targets specific tracts in the brain that likely mediate the immediate effects.](https://med.emory.edu/education/vme/assets/images/gallery/dbs-depression/dbs-for-major-depression.jpg)
<p align="center">DBS of the SCCwm targets specific tracts in the brain that likely mediate the immediate effects. Source: https://med.emory.edu/education/vme/pages/gallery/pf-dbs-depression-image.html</p>

## Overview
Deep brain stimulation (DBS) targeted at a patient's specific subcallosal cingulate white matter (SCCwm) alleviates symptoms of depression.
But how this SCCwm-DBS affects the brain, where and how does it change ongoing brain dynamics, remains unclear.
SCCwm-DBS, when stimulation is initiated at the proper SCCwm target, evokes immediate changes in the brain dynamics along certain brain subnetworks.
I'll call this SCCwm-DBS _network action_ - the where, what, and how of its immediate mechanistic effects.
This is the first piece of the larger question: what is the therapeutic mechanism of action of SCCwm-DBS.

In this repository I address the question: what does SCCwm-DBS do immediately to brain dynamics across multiple scales of measurement.

### Requirements
This repository requires the custom library ```DBSpace``` 
* Repository: [link](https://github.com/virati/DBSpace)
* PyPI: [link](https://pypi.org/project/dbspace/)


### Methods (brief)
Using combined SCC-LFP and dense-EEG, I characterize what modalities I see signal changes in, what spatial pattern I see those signal changes, and what the spatiotemporal nature of those signal changes are.
I also take a computational modeling approach to demonstrate that empirical changes in the EEG are aligned with the stimulated tractography, and that a simple linear classifier can differentiate proper target engagement.

## Network Action
### Spatial Modes
[In Prep]

### Temporal Modes [publication](https://www.frontiersin.org/articles/10.3389/fnins.2022.768355/full)
With an eye towards temporal changes, we studied the immediate effects of SCCwm-DBS on the trajectory of whole brain oscillatory activity.

The code for [Dynamic Oscillations...](https://www.frontiersin.org/articles/10.3389/fnins.2022.768355/full) is available in the ```analysis/DOs``` folder.

### Support Model
If you squint your eyes, you should see that the right-side of the EEG demonstrates more changes than the left-side.
Similarly, if you look at the tractography that is most reliably stimulated under the therapeutic OnTarget condition you'll see a central role for the right-CB.

I'll formalize this a bit better by bridging the tractography with dEEG with [The Virtual Brain]() platform.

## References
* ...
