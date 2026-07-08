## Factorial Optimisation and Multi-Rater Validation of a Deep Learning Framework for Brain Lesion Segmentation in Translational Research  ​

Public Snippets Only. This repository intentionally publishes selected fragments (model skeletons and minimal training interfaces) for U-Net, Deep U-Net, RA-UNet, and ResNet-50–based fine-tuning  to support peer review and reproducibility claims.  

### Models Covered (snippet-only)
- U-Net – deeper encoder/decoder scaffold (dropouts, BN, skip wiring withheld)
- RA-UNet– residual attention gate interfaces (attention/gating internals withheld)
- U-Net (ResNet-50 encoder) – fine-tuning interface (decoder & training loop withheld)
- RA-UNet (ResNet-50 encoder) – fine-tuning interface (attention & training loop withheld)


## Baseline Training (Snippet-Only)-Phae-I
- U-Net 

- RA-UNet 

## Fine-tuning (Snippet-Only)
- U-Net (ResNet-50 encoder)

- RA-UNet (ResNet-50 encoder)

# Full Factorial design of experiment-based model development (Phase -II)
Deep U-Net (5-level / 9-layer, Full-Factorial DOE)


# Machine concordance with independent human raters and nnUNET trained models (Phase -III)
Model B* Vs 4-Individual human raters and Model N* (nn-UNET model)

  
# Hybrid model development 
The above optimisation strategy was further extended to develop a hybrid model (Model HB*) and evaluate its cross-species and cross-condition generalizability using rodent stroke and human meningioma MRI datasets. 

Note: Full implementations (data pipeline, complete training loops, augmentation policy, metrics, callbacks incl. ECE, and evaluation) are withheld and available upon request under the Custom Research License.

# Reference 
- Akter, A., Ahmed, S., & Yousuf, M. A. (2024). Brain Tumor Segmentation Dataset. Https://Www.Kaggle.Com/Datasets/Atikaakter11/Brain-Tumor-Segmentation-Dataset.
- An, J., Wendt, L., Wiese, G., Herold, T., Rzepka, N., Mueller, S., Koch, S. P., Hoffmann, C. J., Harms, C., & Boehm-Sturm, P. (2023). Deep learning-based automated lesion segmentation on mouse stroke magnetic resonance images. Scientific Reports, 13(1). https://doi.org/10.1038/s41598-023-39826-8
- Arora, A., Alderman, J. E., Palmer, J., Ganapathi, S., Laws, E., McCradden, M. D., Oakden-Rayner, L., Pfohl, S. R., Ghassemi, M., McKay, F., Treanor, D., Rostamzadeh, N., Mateen, B., Gath, J., Adebajo, A. O., Kuku, S., Matin, R., Heller, K., Sapey, E., … Liu, X. (2023). The value of standards for health datasets in artificial intelligence-based applications. Nature Medicine, 29(11), 2929–2938. https://doi.org/10.1038/s41591-023-02608-w
- Athanasiou, G., Arcos, J. L., & Cerquides, J. (2023). Enhancing Medical Image Segmentation: Ground Truth Optimization through Evaluating Uncertainty in Expert Annotations. Mathematics, 11(17). https://doi.org/10.3390/math11173771
- Cai, Y., Long, Y., Han, Z., Liu, M., Zheng, Y., Yang, W., & Chen, L. (2023). Swin Unet3D: a three-dimensional medical image segmentation network combining vision transformer and convolution. BMC Medical Informatics and Decision Making, 23(1). https://doi.org/10.1186/s12911-023-02129-z
- Cerqueira, V., Santos, M., Roque, L., Baghoussi, Y., & Soares, C. (2025). Online Data Augmentation for Forecasting with Deep Learning. http://arxiv.org/abs/2404.16918
- Chen, J., Mei, J., Li, X., Lu, Y., Yu, Q., Wei, Q., Luo, X., Xie, Y., Adeli, E., Wang, Y., Lungren, M. P., Zhang, S., Xing, L., Lu, L., Yuille, A., & Zhou, Y. (2024). TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers. Medical Image Analysis, 97. https://doi.org/10.1016/j.media.2024.103280
- Choudhury Avishek and Asan, O. (2020). Role of Artificial Intelligence in Patient Safety Outcomes: Systematic Literature Review. JMIR Med Inform, 8(7), e18599. https://doi.org/10.2196/18599
- Cole, D. J., Drummond, J. C., Ghazal, E. A., & Shapiro, H. M. (1990). A reversible component of cerebral injury as identified by the histochemical stain 2,3,5-triphenyltetrazolium chloride (TTC). Acta Neuropathologica, 80(2), 152–155. https://doi.org/10.1007/BF00308918
- Davila Delgado, J. M., & Oyedele, L. (2021). Deep learning with small datasets: using autoencoders to address limited datasets in construction management. Applied Soft Computing, 112. https://doi.org/10.1016/j.asoc.2021.107836
- De Feo, R., Shatillo, A., Sierra, A., Valverde, J. M., Gröhn, O., Giove, F., & Tohka, J. (2021). Automated joint skull-stripping and segmentation with Multi-Task U-Net in large mouse brain MRI databases. NeuroImage, 229. https://doi.org/10.1016/j.neuroimage.2021.117734
- El-Taraboulsi, J., Cabrera, C. P., Roney, C., & Aung, N. (2023). Deep neural network architectures for cardiac image segmentation. Artificial Intelligence in the Life Sciences, 4. https://doi.org/10.1016/j.ailsci.2023.100083
- Felfeliyan, B., Hareendranathan, A., Kuntze, G., Wichuk, S., Forkert, N. D., Jaremko, J. L., & Ronsky, J. L. (2022). Weakly Supervised Medical Image Segmentation With Soft Labels and Noise Robust Loss. https://doi.org/10.1007/978-3-031-37742-6_47
- Hao, R., Namdar, K., Liu, L., Haider, M. A., & Khalvati, F. (2021). A Comprehensive Study of Data Augmentation Strategies for Prostate Cancer Detection in Diffusion-Weighted MRI Using Convolutional Neural Networks. Journal of Digital Imaging, 34(4), 862–876. https://doi.org/10.1007/s10278-021-00478-7
- Huang, Y., Leotta, N. J., Hirsch, L., Gullo, R. Lo, Hughes, M., Reiner, J., Saphier, N. B., Myers, K. S., Panigrahi, B., Ambinder, E., Di Carlo, P., Grimm, L. J., Lowell, D., Yoon, S., Ghate, S. V., Parra, L. C., & Sutton, E. J. (2025). Cross-site Validation of AI Segmentation and Harmonization in Breast MRI. Journal of Imaging Informatics in Medicine, 38(3), 1642–1652. https://doi.org/10.1007/s10278-024-01266-9
- Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18(2), 203–211. https://doi.org/10.1038/s41592-020-01008-z
- Kamel, P., Kanhere, A., Kulkarni, P., Khalid, M., Steger, R., Bodanapally, U., Gandhi, D., Parekh, V., & Yi, P. H. (2025). Optimizing Acute Stroke Segmentation on MRI Using Deep Learning: Self-Configuring Neural Networks Provide High Performance Using Only DWI Sequences. Journal of Imaging Informatics in Medicine, 38(2), 717–726. https://doi.org/10.1007/s10278-024-00994-2
- Karimi, D., Warfield, S. K., & Gholipour, A. (2021). Transfer learning in medical image segmentation: New insights from analysis of the dynamics of model parameters and learned representations. Artificial Intelligence in Medicine, 116. https://doi.org/10.1016/j.artmed.2021.102078
- Kim, Y., Hrncir, H., Meyer, C. E., Tabbaa, M., Moats, R. A., Levitt, P., Harris, N. G., MacKenzie-Graham, A., & Shattuck, D. W. (2024). Mouse Brain Extractor: Brain segmentation of mouse MRI using global positional encoding and SwinUNETR. https://doi.org/10.1101/2024.09.03.611106
- Koçak, B., Ponsiglione, A., Stanzione, A., Bluethgen, C., Santinha, J., Ugga, L., Huisman, M., Klontzas, M. E., Cannella, R., & Cuocolo, R. (2025). Bias in artificial intelligence for medical imaging: fundamentals, detection, avoidance, mitigation, challenges, ethics, and prospects. In Diagnostic and Interventional Radiology (Vol. 31, Number 2, pp. 75–88). Galenos Publishing House. https://doi.org/10.4274/dir.2024.242854
- Li, X., Morgan, P. S., Ashburner, J., Smith, J., & Rorden, C. (2016). The first step for neuroimaging data analysis: DICOM to NIfTI conversion. Journal of Neuroscience Methods, 264, 47–56. https://doi.org/10.1016/j.jneumeth.2016.03.001
- Liu, X., Ono, K., & Bise, R. (2024). A data augmentation approach that ensures the reliability of foregrounds in medical image segmentation. Image and Vision Computing, 147. https://doi.org/10.1016/j.imavis.2024.105056
- Lyden, P. D., Bosetti, F., Diniz, M. A., Rogatko, A., Koenig, J. I., Lamb, J., Nagarkatti, K. A., Cabeen, R. P., Hess, D. C., Kamat, P. K., Khan, M. B., Wood, K., Dhandapani, K., Arbab, A. S., Leira, E. C., Chauhan, A. K., Dhanesha, N., Patel, R. B., Kumskova, M., … Goh, A. (2022). The Stroke Preclinical Assessment Network: Rationale, Design, Feasibility, and Stage 1 Results. Stroke, 53(5), 1802–1812. https://doi.org/10.1161/STROKEAHA.121.038047
- Mahajan, K. R., & Ontaneda, D. (2017). The Role of Advanced Magnetic Resonance Imaging Techniques in Multiple Sclerosis Clinical Trials. In Neurotherapeutics (Vol. 14, Number 4, pp. 905–923). Springer New York LLC. https://doi.org/10.1007/s13311-017-0561-8
- Milidonis, X., Marshall, I., Macleod, M. R., & Sena, E. S. (2015). Magnetic resonance imaging in experimental stroke and comparison with histology: Systematic review and meta-analysis. Stroke, 46(3), 843–851. https://doi.org/10.1161/STROKEAHA.114.007560
- Morais, A., Imai, T., Jin, X., Locascio, J. J., Boisserand, L., Herman, A. L., Chauhan, A., Lamb, J., Nagarkatti, K., Diniz, M. A., Kumskova, M., Dhanesha, N., Kamat, P. K., Khan, M. B., Dhandapani, K. M., Patel, R. B., Sutariya, B., Shi, Y., van Leyen, K., … Ayata, C. (2024). Biological and Procedural Predictors of Outcome in the Stroke Preclinical Assessment Network (SPAN) Trial. Circulation Research, 135(5), 575–592. https://doi.org/10.1161/CIRCRESAHA.123.324139
- Ni, R., Han, K., Haibe-Kains, B., & Rink, A. (2024). Generalizability of deep learning in organ-at-risk segmentation: A transfer learning study in cervical brachytherapy. Radiotherapy and Oncology, 197. https://doi.org/10.1016/j.radonc.2024.110332
-Nouman, M., Mabrok, M., & Rashed, E. A. (2024). Neuro-TransUNet: Segmentation of stroke lesion in MRI using transformers. http://arxiv.org/abs/2406.06017
- Pradeep, A. (2021). Brain MRI. Https://Www.Kaggle.Com/Datasets/Pradeep2665/Brain-Mri.
- Razavi, M., Mavaddati, S., & Koohi, H. (2024). ResNet deep models and transfer learning technique for classification and quality detection of rice cultivars. Expert Systems with Applications, 247. https://doi.org/10.1016/j.eswa.2024.123276
- Renard, F., Guedria, S., Palma, N. De, & Vuillerme, N. (2020). Variability and reproducibility in deep learning for medical image segmentation. Scientific Reports, 10(1), 13724. https://doi.org/10.1038/s41598-020-69920-0
- Skorupko, G., Avgoustidis, F., Martín-Isla, C., Garrucho, L., Kessler, D. A., Pujadas, E. R., Díaz, O., Bobowicz, M., Gwoździewicz, K., Bargalló, X., Jarus̆evic̆ius, P., Osuala, R., Kushibar, K., & Lekadir, K. (2025). Federated nnU-Net for privacy-preserving medical image segmentation. Scientific Reports, 15(1). https://doi.org/10.1038/s41598-025-22239-0
- Sylolypavan, A., Sleeman, D., Wu, H., & Sim, M. (2023). The impact of inconsistent human annotations on AI driven clinical decision making. Npj Digital Medicine, 6(1). https://doi.org/10.1038/s41746-023-00773-3
- Tayebi Arasteh, S., Kuhl, C., Saehn, M. J., Isfort, P., Truhn, D., & Nebelung, S. (2023). Enhancing domain generalization in the AI-based analysis of chest radiographs with federated learning. Scientific Reports, 13(1). https://doi.org/10.1038/s41598-023-49956-8
- Tummala, B. M., Jaladi, R., Veeraiah, D. C., & Peruri, A. K. (2024). TransUNet for Precise and Robust GI tract Segmentation in MRI Images. Journal of Image and Graphics (United Kingdom), 12(3), 302–311. https://doi.org/10.18178/joig.12.3.302-311
- Walid null, E. null, Lina null, H. null, & Yongmin null, L. null. (2024). UNet and Variants for Medical Image Segmentation. International Journal of Network Dynamics and Intelligence, 3(2), 100009. https://doi.org/10.53941/ijndi.2024.100009
- Werdiger, F., Yogendrakumar, V., Visser, M., Kolacz, J., Lam, C., Hill, M., Chen, C., Parsons, M. W., & Bivard, A. (2024). Clinical performance review for 3-D Deep Learning segmentation of stroke infarct from diffusion-weighted images. Neuroimage: Reports, 4(1). https://doi.org/10.1016/j.ynirp.2024.100196
- Xiao, R., Li, Z., Miao, X., Wang, W., & Zhang, P. (2022). GuidedMix: An on-the-fly data augmentation approach for robust speaker recognition system. In Electronics Letters (Vol. 58, Number 2, pp. 82–85). John Wiley and Sons Inc. https://doi.org/10.1049/ell2.12354
- Xiao, S., Wang, P., Diao, W., Rong, X., Li, X., Fu, K., & Sun, X. (2023). MoCG: Modality Characteristics-Guided Semantic Segmentation in Multimodal Remote Sensing Images. IEEE Transactions on Geoscience and Remote Sensing, 61, 1–18. https://doi.org/10.1109/TGRS.2023.3334471
- Ying, X. (2019). An Overview of Overfitting and its Solutions. Journal of Physics: Conference Series, 1168(2), 022022. https://doi.org/10.1088/1742-6596/1168/2/022022
- Yu, X., Zhao, H., Zhang, M., Wei, Y., Zhou, L., & Ou, L. (2024). DynamicAug: Enhancing Transfer Learning Through Dynamic Data Augmentation Strategies Based on Model State. Neural Processing Letters, 56(3). https://doi.org/10.1007/s11063-024-11626-9
- Zhou, S. K., Greenspan, H., Davatzikos, C., Duncan, J. S., Van Ginneken, B., Madabhushi, A., Prince, J. L., Rueckert, D., & Summers, R. M. (2021). A Review of Deep Learning in Medical Imaging: Imaging Traits, Technology Trends, Case Studies with Progress Highlights, and Future Promises. Proceedings of the IEEE, 109(5), 820–838. https://doi.org/10.1109/JPROC.2021.3054390
- Zille, M., Farr, T. D., Przesdzing, I., Müller, J., Sommer, C., Dirnagl, U., & Wunder, A. (2012). Visualizing cell death in experimental focal cerebral ischemia: Promises, problems, and perspectives. In Journal of Cerebral Blood Flow and Metabolism (Vol. 32, Number 2, pp. 213–231). https://doi.org/10.1038/jcbfm.2011.150
 


