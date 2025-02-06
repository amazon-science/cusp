## Spectro-Riemannian Graph Neural Networks
Official code repository for [Spectro-Riemannian Graph Neural Networks](https://www.arxiv.org/abs/2502.00401) (ICLR 2025).


## 🛠 Dependencies and Installation

- [geoopt](https://github.com/geoopt/geoopt) `0.5.0` (For Riemannian optimization and algebra)
- [GraphRicciCurvature](https://github.com/saibalmars/GraphRicciCurvature) `0.5.3.2` (For computing Ollivier-Ricci Curvature)
- torch `2.4.0` (Implementation in PyTorch)
- torch_geometric `2.6.1`
- Other required packages in `requirements.txt`

```python
# git clone this repository
git clone https://github.com/amazon-science/cusp.git

# install python dependencies
pip3 install -r requirements.txt
```

## 📞 Contact
If you have any questions or issues, please feel free to reach out to [Karish Grover](https://karish-grover.github.io/) at <a href="mailto:karishg@cs.cmu.edu">karishg@cs.cmu.edu</a>.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the CC-BY-NC-4.0 License.

## ✏️ Citation

If you think that this work is helpful, please leave a star ⭐️ and cite our paper:

```
@misc{grover2025spectroriemanniangraphneuralnetworks,
      title={Spectro-Riemannian Graph Neural Networks}, 
      author={Karish Grover and Haiyang Yu and Xiang Song and Qi Zhu and Han Xie and Vassilis N. Ioannidis and Christos Faloutsos},
      year={2025},
      eprint={2502.00401},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.00401}, 
}
```