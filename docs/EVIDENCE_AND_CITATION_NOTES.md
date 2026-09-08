# Evidence and Citation Notes

## Scope

This note maps the study's literature-facing claims to official dataset documentation, peer-reviewed papers, and official conference proceedings available through September 4, 2026. Reference [15] is a contemporaneous preprint and must remain labeled as such. The citations support the framing and comparison logic; they do not substitute for reporting this study's own data audit, code version, patient-disjoint splits, or unrun results.

## Claim-to-source map

| Manuscript claim | Evidence boundary and safe wording | Sources |
|---|---|---|
| LIDC-IDRI preserves reader disagreement rather than imposing one consensus contour. | The original protocol used four thoracic radiologists, a blinded read followed by an unblinded read, and no forced consensus. Each reader could retain, add, delete, or modify a mark. Call the masks **case-local annotation supports**; do not imply a stable reader identity across cases. | [1]–[3] |
| The cumulative targets represent support among four review slots. | Define the voxelwise contour count as `C(v) = sum_r Y_r(v)` for four slots, the empirical support as `q(v) = C(v)/4`, and `T_k(v) = 1[C(v) >= k]`. In this release, an empty higher slot means no clustered contour in that review slot, not an unknown denominator. This last point is an archive-specific provenance decision and must also be supported by the repository audit. | [1]–[3] |
| Label fusion is an established alternative, not the proposed novelty. | STAPLE jointly estimates a probabilistic latent segmentation and each segmentation generator's performance using expectation maximization. It is an appropriate classical fusion baseline, but it produces a fused target rather than four ordered spatial endpoints. | [4] |
| Generative ambiguity modeling is established on LIDC-IDRI. | Probabilistic U-Net learns a conditional distribution over plausible masks with a U-Net/cVAE; PHiSeg adds hierarchical latent variables across spatial scales. These methods generate plausible samples rather than directly segmenting the four cumulative support regions. | [5], [6] |
| Annotator-aware latent-truth and calibration models predate this study. | Tanno *et al.* estimate a latent label distribution with annotator confusion models; MRNet models multi-rater agreement for calibrated segmentation; Zhang *et al.* jointly estimate consensus and image-dependent, pixelwise annotator confusion. These support a multi-rater motivation but do not support a claim that multi-rater modeling itself is new. | [7]–[9] |
| Diffusion-based ambiguous segmentation is a close but different task formulation. | CIMD learns an unordered distribution of plausible segmentations and their occurrence frequencies. DiffOSeg combines population-level probabilistic consensus with expert-prompted personalization. Neither paper defines `T1`–`T4` as four directly evaluated cumulative endpoints. | [10], [13] |
| Union/intersection prediction is a closely related peer-reviewed deterministic LIDC-IDRI precedent. | UGMCS-Net supervises the annotation union, intersection, and a balanced final mask. It establishes that explicit disagreement-region targets are not new; the present distinction is the direct treatment of all four cumulative support regions, including the intermediate `T2` and `T3` regions. | [11] |
| Personalized multi-rater segmentation is established on LIDC-IDRI. | D-Persona combines diversified Probabilistic U-Net outputs with personalized projection heads. Because persistent annotator identities are unavailable in its LIDC-IDRI setup, it constructs preferences by ordering annotations by area. The 2026 CVPR method uses harmonized features, frequency-domain rater prompts, and distributional regularization for personalized/probabilistic outputs. These are not cumulative support-threshold models. | [12], [14] |
| Ordinal vote-count supervision and RPS training are no longer available as a broad novelty claim. | Riera-Marín *et al.* convert `K` binary annotations to an exact count in `{0,...,K}`, predict `K+1` categorical probabilities, and combine BCE with the Ranked Probability Score. The paper evaluates non-LIDC datasets. Treat the repository's `A7_ordinal_rps` as an architecture-matched comparison to this loss formulation, not as a reproduction of its published experiments. | [15] |
| A 2.5D pulmonary-nodule U-Net is established. | Ni *et al.* use 3-D localization followed by 2.5-D multiscale U-Net refinement on LIDC-IDRI. Therefore, neither adjacent-slice input nor a 2.5-D U-Net should be claimed as independently novel. | [16] |

## Narrow novelty language

### Recommended statement

> Within the peer-reviewed LIDC-IDRI methods reviewed here, this study evaluates a direct cumulative agreement-region formulation for candidate-centered pulmonary-nodule segmentation. A single 2.5-D residual encoder-decoder predicts four case-local support masks, `T_k(v) = 1[C(v) >= k]` for `k = 1,...,4`, while guaranteeing their voxelwise nesting by construction. In contrast to methods that estimate a single fused mask, sample plausible masks, personalize predictions, or model only union and intersection, the study directly trains and evaluates all four nested support regions. The claimed contribution is this task formulation, exact nested output parameterization, and controlled evaluation—particularly of the intermediate `T2` and `T3` regions—not the backbone, 2.5-D context, multi-rater supervision, or ordinal modeling in isolation.

### Permissible literature-review sentence

> In the peer-reviewed LIDC-IDRI methods reviewed in [5], [6], and [9]–[14], outputs take the form of fused consensus estimates, stochastic samples, personalized masks, or union/intersection regions; these studies do not directly evaluate all four cumulative support masks as nested segmentation endpoints.

Keep the qualifier **“reviewed”** and the named citation scope. This is a bounded literature comparison, not an absolute priority claim.

### Claims to avoid

- “The first multi-rater lung-nodule segmentation model.”
- “The first ordinal multi-rater segmentation method.”
- “The first 2.5-D U-Net for pulmonary-nodule segmentation.”
- “The first method to model annotator disagreement, uncertainty, union/intersection, or soft vote fractions.”
- “Reader-specific” or “radiologist-specific” predictions for this release; use **case-local annotation support** unless persistent reader identity is actually available.

## Exact ordinal comparator note

For `K` raters, Riera-Marín *et al.* [15] define the ordinal consensus label and its predicted categorical distribution as

\[
\tilde y(v)=\sum_{r=1}^{K} y^{(r)}(v)\in\{0,\ldots,K\},
\qquad
\hat p_k(v)\approx P(\tilde y(v)=k).
\]

With one-hot count target `y_k(v)`, the paper uses

\[
L_{\mathrm{RPS}}=
\frac{1}{K+1}\sum_{j=0}^{K}
\left(F_j(v)-\hat F_j(v)\right)^2,
\quad
F_j=\sum_{k=0}^{j}y_k,
\quad
\hat F_j=\sum_{k=0}^{j}\hat p_k,
\]

and maps the count distribution to a binary foreground probability with

\[
\hat p_{\mathrm{fg}}(v)=\sum_{k\ge K/2}^{K}\hat p_k(v).
\]

Its final objective is `L = L_BCE + alpha L_RPS`, with `alpha = 0.8` selected on validation data. The reported implementation uses a pretrained ResNet encoder and Feature Pyramid Network decoder and evaluates CoCaHis, REFUGE-Disc, REFUGE-Cup, and LongCIU with MR-ECE and AUC. For `K = 4`, an exact-count model also induces coherent cumulative probabilities through `P(C >= k) = 1 - P(C <= k - 1)`. Accordingly, `A7_ordinal_rps` is a necessary comparator for output/loss parameterization; it does not remove the narrower contribution of directly optimizing and evaluating the four spatial support regions.

## Numbered references

[1] The Cancer Imaging Archive, “LIDC-IDRI,” National Cancer Institute, collection page. [Online]. Available: https://www.cancerimagingarchive.net/collection/lidc-idri/

[2] S. G. Armato III *et al.*, “The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans,” *Medical Physics*, vol. 38, no. 2, pp. 915–931, 2011, doi: 10.1118/1.3528204. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC3041807/

[3] M. F. McNitt-Gray *et al.*, “The Lung Image Database Consortium (LIDC) data collection process for nodule detection and annotation,” *Academic Radiology*, vol. 14, no. 12, pp. 1464–1474, 2007, doi: 10.1016/j.acra.2007.07.021. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC2176079/

[4] S. K. Warfield, K. H. Zou, and W. M. Wells, “Simultaneous Truth and Performance Level Estimation (STAPLE): An algorithm for the validation of image segmentation,” *IEEE Transactions on Medical Imaging*, vol. 23, no. 7, pp. 903–921, 2004, doi: 10.1109/TMI.2004.828354. [Online]. Available: https://pubmed.ncbi.nlm.nih.gov/15250643/

[5] S. Kohl *et al.*, “A Probabilistic U-Net for segmentation of ambiguous images,” in *Advances in Neural Information Processing Systems 31*, 2018. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2018/hash/473447ac58e1cd7e96172575f48dca3b-Abstract.html

[6] C. F. Baumgartner *et al.*, “PHiSeg: Capturing uncertainty in medical image segmentation,” in *Medical Image Computing and Computer Assisted Intervention—MICCAI 2019*, LNCS 11765, pp. 119–127, 2019, doi: 10.1007/978-3-030-32245-8_14. [Online]. Available: https://link.springer.com/chapter/10.1007/978-3-030-32245-8_14

[7] R. Tanno, A. Saeedi, S. Sankaranarayanan, D. C. Alexander, and N. Silberman, “Learning from noisy labels by regularized estimation of annotator confusion,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2019. [Online]. Available: https://openaccess.thecvf.com/content_CVPR_2019/html/Tanno_Learning_From_Noisy_Labels_by_Regularized_Estimation_of_Annotator_Confusion_CVPR_2019_paper.html

[8] W. Ji *et al.*, “Learning calibrated medical image segmentation via multi-rater agreement modeling,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2021, doi: 10.1109/CVPR46437.2021.01216. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2021/html/Ji_Learning_Calibrated_Medical_Image_Segmentation_via_Multi-Rater_Agreement_Modeling_CVPR_2021_paper.html

[9] L. Zhang *et al.*, “Learning from multiple annotators for medical image segmentation,” *Pattern Recognition*, vol. 138, Art. no. 109400, 2023, doi: 10.1016/j.patcog.2023.109400. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC10533416/

[10] A. Rahman, J. M. J. Valanarasu, I. Hacihaliloglu, and V. M. Patel, “Ambiguous medical image segmentation using diffusion models,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2023, doi: 10.1109/CVPR52729.2023.01110. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2023/html/Rahman_Ambiguous_Medical_Image_Segmentation_Using_Diffusion_Models_CVPR_2023_paper.html

[11] H. Yang *et al.*, “Lung nodule segmentation and uncertain region prediction with an uncertainty-aware attention mechanism,” *IEEE Transactions on Medical Imaging*, vol. 43, no. 4, pp. 1284–1295, 2024, doi: 10.1109/TMI.2023.3332944. [Online]. Available: https://ieeexplore.ieee.org/document/10318829/

[12] Y. Wu *et al.*, “Diversified and personalized multi-rater medical image segmentation,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 11470–11479, 2024, doi: 10.1109/CVPR52733.2024.01090. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Diversified_and_Personalized_Multi-rater_Medical_Image_Segmentation_CVPR_2024_paper.html

[13] H. Zhang, X. Luo, Y. Chen, and K. Li, “DiffOSeg: Omni medical image segmentation via multi-expert collaboration diffusion model,” in *Medical Image Computing and Computer Assisted Intervention—MICCAI 2025*, LNCS 15972, pp. 128–138, 2025, doi: 10.1007/978-3-032-05169-1_13. [Online]. Available: https://papers.miccai.org/miccai-2025/0233-Paper5223.html

[14] S. Karimijafarbigloo, A. Khosravi, A. Kheyrkhah, R. Azad, M. Reyes, and D. Merhof, “Harmonized feature conditioning and frequency-prompt personalization for multi-rater medical segmentation,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2026. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2026/html/Karimijafarbigloo_Harmonized_Feature_Conditioning_and_Frequency-Prompt_Personalization_for_Multi-Rater_Medical_Segmentation_CVPR_2026_paper.html

[15] M. Riera-Marín, J. García López, J. Rodríguez-Comas, M. A. González Ballester, and A. Galdran, “Multi-Rater Calibrated Segmentation Models,” arXiv:2605.02437v1, May 2026, **preprint**. [Online]. Available: https://arxiv.org/html/2605.02437v1

[16] Y. Ni, Z. Xie, D. Zheng, Y. Yang, and W. Wang, “Two-stage multitask U-Net construction for pulmonary nodule segmentation and malignancy risk prediction,” *Quantitative Imaging in Medicine and Surgery*, vol. 12, no. 1, pp. 292–309, 2022, doi: 10.21037/qims-21-19. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC8666775/
