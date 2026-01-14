# AutoBrep: Autoregressive B-Rep Generation with Unified Topology and Geometry (SIGGRAPH Asia 2025)

[![arXiv](https://img.shields.io/badge/📃-arXiv%20-red.svg)](https://arxiv.org/abs/2512.03018) 
[![webpage](https://img.shields.io/badge/🌐-Website%20-blue.svg)]() 

[Xiang Xu*](https://samxuxiang.github.io/), [Pradeep Jayaraman*](https://www.research.autodesk.com/people/pradeep-kumar-jayaraman/), [Joseph Lambourne*](https://www.research.autodesk.com/people/joseph-george-lambourne/), [Yilin Liu](https://yilinliu77.github.io/), [Durvesh Malpure](https://www.research.autodesk.com/people/durvesh-malpure/), [Pete Meltzer](https://www.research.autodesk.com/people/pete-meltzer/)

![alt](https://samxuxiang.github.io/img/autobrep_teaser.png)

> AutoBrep is a decoder-only Transformer model that autoregressively generates B-Rep geometry and topology tokens following a BFS order of the B-Rep topology graph. Geometric information is tokenized as bounding boxes paired with encoded UV-grid shape codes. Topological structure is represented via a special face identifier that maps face–edge adjacencies into reference tokens.


### Installing environments

```
cd core
conda env create -f dev-env.yaml
conda activate autobrep
```

### Inference

```
sh scripts/sample.sh
```

Download the [pretrained checkpoints](https://huggingface.co/SamGiantEagle/AutoBrep)

Modify ```configs/sample.json``` for different sampling parameters. 

```complexity``` can be random, easy, medium, or hard. 



### Training

```
sh scripts/train.sh
```

Download the [deduplicated ABC-1M dataset](https://huggingface.co/datasets/SamGiantEagle/ABC-1M)

Modify ```configs/autobrep.yaml``` with the path to ABC parquets and pretrained FSQ checkpoints.


### Citation
```
@inproceedings{xu2025autobrep,
  title={AutoBrep: Autoregressive B-Rep Generation with Unified Topology and Geometry},
  author={Xu, Xiang and Jayaraman, Pradeep and Lambourne, Joseph and Liu, Yilin and Malpure, Durvesh and Meltzer, Pete},
  booktitle={Proceedings of the SIGGRAPH Asia 2025 Conference Papers},
  pages={1--12},
  year={2025}
}
```
