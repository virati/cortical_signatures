# Cortical Signatures of SCCwm-DBS
## Overview
Deep brain stimulation (DBS) is proving to be an effective treatment for severe, treatment resistant depression (TRD).
One of the most well-studied targets for DBS is the subcallosal cingulate cortex (SCC), and demonstrations of SCC-DBS have yielded positive \cite{} but equivocal \cite{} results.

![DBS of the SCCwm targets specific tracts in the brain that likely mediate the immediate effects.](https://med.emory.edu/education/vme/assets/images/gallery/dbs-depression/dbs-for-major-depression.jpg)
<p align="center">DBS of the SCCwm targets specific tracts in the brain that likely mediate the immediate effects. Source: https://med.emory.edu/education/vme/pages/gallery/pf-dbs-depression-image.html</p>

Precise stimulation of the subcallosal cingulate white matter (SCCwm) is now thought to be necessary to achieve antidepressant effect when implementing SCC-DBS.
However, we need objective signatures to confirm, study, and engineer around antidepressant SCCwm-DBS.
Neural recordings are a great modality to derive and use these objective signatures.

## This Repository
This repository contains all the code needed to regenerate the figures from Chapter ? from my [dissertation]().
This chapter is focused on characterized the direct effects of SCCwm-DBS across whole-brain networks - its *network action*.

### Requirements
This repository requires the custom library ```DBSpace``` 
* Repository: [link](https://github.com/virati/DBSpace)
* PyPI: [link](https://pypi.org/project/dbspace/)


## Publications

### Spatial Modes
Preprint: [Network Action of...](https://www.medrxiv.org/content/10.1101/2022.07.27.22278130v1)

First, I wanted to understand *where* in the brain we would see changes - locally around the SCCwm target, remotely in scalp-measurable cortex, some combination of both, or neither.
This gives us the spatial modes of the network action.

### Temporal Modes
Pub: [Dynamic Oscillations...](https://www.frontiersin.org/articles/10.3389/fnins.2022.768355/full)

With an eye towards temporal changes, we studied the immediate effects of SCCwm-DBS on the trajectory of wide-brain oscillatory activity.

The code for this publication is in the ```analysis/DOs``` folder.

## References
* ...
