import os
import base64
import subprocess

BASE_DIR = '/Users/anilkumar/Downloads/ORCHID/paper'
IMG_DIR = os.path.join(BASE_DIR, 'extracted_images')

def img_to_base64(path):
    with open(path, 'rb') as f:
        data = f.read()
    ext = 'png' if path.endswith('.png') else 'jpeg'
    return f"data:image/{ext};base64,{base64.b64encode(data).decode('utf-8')}"

# Load images
fig1 = img_to_base64(os.path.join(IMG_DIR, 'p3_img25.png'))
fig2 = img_to_base64(os.path.join(IMG_DIR, 'p3_img23.png'))
fig3 = img_to_base64(os.path.join(IMG_DIR, 'p3_img26.jpeg'))
fig4 = img_to_base64(os.path.join(IMG_DIR, 'p4_img29.png'))
fig5 = img_to_base64(os.path.join(IMG_DIR, 'p4_img30.png'))
fig6 = img_to_base64(os.path.join(IMG_DIR, 'p4_img31.png'))
fig7 = img_to_base64(os.path.join(IMG_DIR, 'p4_img33.png'))
fig8 = img_to_base64(os.path.join(IMG_DIR, 'p5_img38.png'))
fig9 = img_to_base64(os.path.join(IMG_DIR, 'p5_img39.png'))
fig10 = img_to_base64(os.path.join(IMG_DIR, 'p5_img41.png'))
fig11 = img_to_base64(os.path.join(IMG_DIR, 'p5_img40.png'))
fig12 = img_to_base64(os.path.join(IMG_DIR, 'p6_img44.jpeg'))
fig13 = img_to_base64(os.path.join(IMG_DIR, 'p6_img45.png'))
fig14 = img_to_base64(os.path.join(IMG_DIR, 'p6_img46.png'))

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Conditional StyleGAN2-ADA for Synthetic Oral Histopathology Image Generation: A Pure-Synthetic Downstream Classification Evaluation on the ORCHID Dataset</title>
    <style>
        @page {{
            size: letter;
            margin: 0.75in 0.63in 1in 0.63in;
        }}
        body {{
            font-family: 'Times New Roman', Times, serif;
            font-size: 10pt;
            line-height: 1.15;
            color: #000;
            background: #fff;
            margin: 0 auto;
            max-width: 8.5in;
            padding: 0.5in 0.63in;
            box-sizing: border-box;
        }}
        .header {{
            text-align: center;
            margin-bottom: 18pt;
        }}
        h1.title {{
            font-size: 20pt;
            font-weight: bold;
            line-height: 1.2;
            margin: 0 0 10pt 0;
        }}
        .authors {{
            font-size: 11pt;
            font-weight: bold;
            margin-bottom: 4pt;
        }}
        .affil {{
            font-size: 9pt;
            font-style: italic;
            color: #222;
            margin-bottom: 16pt;
        }}
        .content-columns {{
            column-count: 2;
            column-gap: 0.25in;
            column-rule: 0.5pt solid #eee;
            text-align: justify;
        }}
        .abstract-container {{
            margin-bottom: 8pt;
            font-size: 9pt;
            text-align: justify;
            text-indent: 14pt;
        }}
        .abstract-label {{
            font-weight: bold;
            font-style: italic;
        }}
        .abstract-text {{
            font-weight: bold;
        }}
        .keywords-container {{
            margin-bottom: 12pt;
            font-size: 9pt;
            text-align: justify;
            text-indent: 14pt;
        }}
        .keywords-label {{
            font-weight: bold;
            font-style: italic;
        }}
        .keywords-text {{
            font-style: italic;
        }}
        h2.sec-heading {{
            font-size: 10pt;
            font-weight: bold;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 0.5pt;
            margin-top: 14pt;
            margin-bottom: 4pt;
            break-after: avoid;
        }}
        h3.subsec-heading {{
            font-size: 10pt;
            font-weight: normal;
            font-style: italic;
            text-align: left;
            margin-top: 8pt;
            margin-bottom: 3pt;
            break-after: avoid;
        }}
        p {{
            margin: 0;
            text-indent: 14pt;
            text-align: justify;
        }}
        .equation {{
            text-align: center;
            margin: 6pt 0;
            font-style: italic;
            text-indent: 0;
        }}
        .equation span.eq-num {{
            float: right;
            font-style: normal;
        }}
        .figure {{
            margin: 8pt 0;
            text-align: center;
            break-inside: avoid;
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
            border: 0.5pt solid #ddd;
        }}
        .caption {{
            font-size: 8pt;
            line-height: 1.1;
            text-align: justify;
            margin-top: 3pt;
            text-indent: 0;
        }}
        .caption-bold {{
            font-weight: bold;
        }}
        .table-container {{
            margin: 8pt 0;
            break-inside: avoid;
            text-align: center;
        }}
        .table-title {{
            font-size: 8pt;
            font-weight: bold;
            text-transform: uppercase;
            margin-bottom: 2pt;
        }}
        table.ieee-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 7.5pt;
            line-height: 1.1;
            margin: 0 auto;
        }}
        table.ieee-table th, table.ieee-table td {{
            padding: 3pt 4pt;
            text-align: left;
        }}
        table.ieee-table thead tr:first-child {{
            border-top: 1.5pt solid #000;
            border-bottom: 0.75pt solid #000;
        }}
        table.ieee-table tbody tr:last-child {{
            border-bottom: 1.5pt solid #000;
        }}
        table.ieee-table th {{
            font-weight: bold;
            text-align: center;
        }}
        .ref-item {{
            font-size: 8pt;
            line-height: 1.15;
            margin-bottom: 2pt;
            text-align: justify;
            text-indent: -18pt;
            padding-left: 18pt;
        }}
    </style>
</head>
<body>

<div class="header">
    <h1 class="title">Conditional StyleGAN2-ADA for Synthetic Oral Histopathology Image Generation: A Pure-Synthetic Downstream Classification Evaluation on the ORCHID Dataset</h1>
    <div class="authors">Chakkilala Anil Kumar &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Malladi Nagasri &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Bandaru Keerthana</div>
    <div class="affil">Department of Computer Science and Engineering<br>ORCHID Oral Histopathology Benchmark Study</div>
</div>

<div class="content-columns">

    <div class="abstract-container">
        <span class="abstract-label">Abstract—</span>
        <span class="abstract-text">Oral Squamous Cell Carcinoma (OSCC) and its premalignant precursor, Oral Submucous Fibrosis (OSF), pose a severe public health burden in South Asia; however, geographic data disparities and severe class imbalances significantly undermine the diagnostic accuracy of medical AI models. Aiming to overcome this scarcity, we explored the clinical viability of generating histopathology images using a conditional StyleGAN2-ADA model trained on the ORCHID dataset. In spite of achieving an impressive Fréchet Inception Distance (FID) of 6.98, comprehensive downstream testing exposed a substantial discrepancy between visual fidelity and downstream diagnostic utility. In synthetic-only training regimes, classifiers (DenseNet-121 and EfficientNet-B3) achieved below 10% accuracy when tested against authentic clinical biopsies, indicating a failure to encode fine-grained microscopic features. Subsequent data-mixing experiments revealed that while conservative synthetic supplementation effectively balances minority classes without compromising diagnostic accuracy, excessive synthetic saturation actively dilutes genuine biological signals, resulting in performance decline. Overall, this study highlights the clinical deception of purely visual realism; the results support using synthetic data conservatively as a complementary augmentation strategy rather than as an unconstrained replacement for real patient tissue.</span>
    </div>

    <div class="keywords-container">
        <span class="keywords-label">Index Terms—</span>
        <span class="keywords-text">Oral Histopathology, StyleGAN2-ADA, ORCHID Dataset, Synthetic Data Generalization, Pure-Synthetic Stress Test, DenseNet-121, EfficientNet-B3.</span>
    </div>

    <h2 class="sec-heading">I. Introduction</h2>
    <p>In South Asia, the disproportionate burden of cancer-related morbidity and mortality is driven significantly by oral cancer, particularly Oral Squamous Cell Carcinoma (OSCC) and its high-risk precursor state, Oral Submucous Fibrosis (OSF). The etiology of oral malignancies in South Asia is strongly linked to habit-related carcinogens, such as consumption of betel quid, gutka, and smokeless tobacco, consequently yielding distinct genomic, inflammatory, and structural phenotypes. Despite this immense disease burden, clinical translation in computational pathology is bottlenecked by profound geographic representation bias. Since more than half of global clinical AI datasets originate from Western or East Asian centres, neural networks frequently exploit non-clinical shortcuts instead of learning valid histological features. As a result, systems trained on foreign datasets routinely experience catastrophic domain generalization failures when deployed in Indian clinical environments.</p>
    <p>To mitigate this representation gap, the multi-center ORCHID dataset was established, compiling diverse H&E oral histopathology images across five diagnostic categories. Yet, consistent with clinical data collection, the ORCHID dataset contains moderate class imbalances. Consequently, this reduces the model sensitivity on minority, high-complexity classes, including normal oral mucosa and aggressive PDOSCC. To address this data bottleneck, generative modelling has emerged as a promising countermeasure. Generative Adversarial Networks (GANs), specifically StyleGAN2-ADA, offer a promising framework to synthesize high-resolution tissue patches and balance underrepresented clinical cohorts.</p>
    <p>However, an essential question persists in medical AI: Does visual and statistical fidelity reliably translate into clinical diagnostic utility? Standard metrics like Fréchet Inception Distance (FID) rely on general-purpose feature extractors rather than microscopic diagnostic features. Focusing on global color and broad textures, these metrics are easily misled by visually plausible synthetic tissue lacking essential cellular atypia and nuclear pleomorphism.</p>
    <p>In this work, we investigate the divergence between statistical generative realism and clinical diagnostic utility through a rigorous, multi-stage empirical framework. We initially optimized a class-conditional StyleGAN2-ADA model on the ORCHID dataset to generate a benchmark repository of high-fidelity synthetic patches. To transcend standard statistical metrics, we evaluate downstream diagnostic validity using DenseNet-121 and EfficientNet-B3 architectures across three distinct experiments. We first tested classifiers trained purely on 25,000 synthetic images, followed by a controlled mixing setup where minority classes were supplemented to match majority-class size. Finally, we performed scale-expansion experiments by padding each diagnostic class up to 5,000 images. We applied the same α-controlled mixing formulation to govern the integration of real and synthetic histopathology patches.</p>

    <h2 class="sec-heading">II. Related Work</h2>
    <p>Deep convolutional architectures such as DenseNet-121 [1] and EfficientNet-B3 [2] serve as standard backbones for fine-grained medical image classification. Despite their efficacy, the translation of these networks to South Asian oral oncology has historically been constrained by severe regional data deficits [3]. The dataset consists of approx. 300,000 raw high-resolution H&E-stained patches at 1000× effective magnification, and the final curated benchmark encompasses 14,705 images classified into five diagnostic categories: normal, OSMF, WDOSCC, MDOSCC, and PDOSCC [4]. Reflecting routine clinical intake, the dataset displays a moderate class imbalance ratio of approximately 2.67:1 [4], [5]. While WDOSCC and MDOSCC heavily dominate the distribution, normal tissue and aggressive PDOSCC cases remain severely underrepresented.</p>

    <div class="table-container">
        <div class="table-title">TABLE I<br>ORCHID Dataset Class Distribution</div>
        <table class="ieee-table">
            <thead>
                <tr>
                    <th style="text-align:left;">Diagnostic Class</th>
                    <th style="text-align:right;">Number of Real Images</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Normal Oral Mucosa</td><td style="text-align:right;">1,502</td></tr>
                <tr><td>Oral Submucous Fibrosis (OSMF)</td><td style="text-align:right;">3,015</td></tr>
                <tr><td>Well-Differentiated OSCC (WDOSCC)</td><td style="text-align:right;">4,011</td></tr>
                <tr><td>Moderately-Differentiated OSCC (MDOSCC)</td><td style="text-align:right;">3,880</td></tr>
                <tr><td>Poorly-Differentiated OSCC (PDOSCC)</td><td style="text-align:right;">2,297</td></tr>
                <tr style="font-weight:bold; border-top:0.75pt solid #000;"><td>Total Cohort</td><td style="text-align:right;">14,705</td></tr>
            </tbody>
        </table>
    </div>

    <p>Prior studies utilizing the ORCHID benchmark have highlighted this distribution skew as a critical bottleneck limiting downstream diagnostic sensitivity. Previous whole-slide image (WSI) studies emphasized the scarcity of poorly differentiated carcinomas, identifying PDOSCC as a decisive bottleneck for automated grading [6]. To mitigate these skews, prior efforts either discarded majority-class data samples to achieve uniform class representation [7] or restricted the scope to binary classification (Normal versus OSCC) using stratified sampling and class-weighted loss to encounter a 6.78:1 imbalance [8].</p>
    <p>Recently, long-tailed classification methods frame PDOSCC as a vulnerable tail category, implementing complex distribution-aware loss reweighting and tail-specific confidence gating to prevent F1-score collapse [9]. While these computational interventions effectively facilitate optimization convergence, they functionally treat the symptom of data skew through loss re-weighting rather than resolving the core biological dearth of representative tissue phenotypes.</p>
    <p>Seeking to resolve these biological deficits at the data level, Generative Adversarial Networks (GANs) are widely deployed to synthesize high-resolution histopathological imagery [10]. StyleGAN2-ADA is particularly well suited for oral histology synthesis because its style-based generator decouples global architectural patterns from subtle cellular details, and the adaptive discriminator augmentation (ADA) mitigates discriminator overfitting under acute sample scarcity [11]. Prior work has combined StyleGAN2-ADA with transfer learning in data-constrained domains such as breast cancer [11] and thyroid tissue [12], verifying that the architecture synthesizes photorealistic textures even under acute sample constraints.</p>
    <p>Despite these advancements, a pervasive methodological limitation persists in how synthetic histopathology is evaluated, with prevailing literature depending on the Fréchet Inception Distance (FID) as the primary arbiter of generative quality [13]. Because FID operates on natural-image features, it captures macroscopic textures while remaining blind to the fine microscopic markers required for histological grading [14], [15]. Moreover, existing literature evaluates synthetic samples only when combined with real images [15], [16], rather than testing them in isolation. As a result, existing literature stresses the absence of systematic, multimodal stress tests that evaluate synthetic images in complete isolation. Without evaluating the synthetic data in complete isolation, it is impossible to confirm whether low FID values reflect biological validity or merely superficial visual realism.</p>

    <h2 class="sec-heading">III. Methodology: Generative Synthesis and GAN Training</h2>
    <p>To counteract data scarcity and class skew in the ORCHID repository, we deployed StyleGAN2-ADA with Adaptive Discriminator Augmentation (ADA) to stabilize training. To seed the network with generalized histological feature distributions, the model was initialized using BreCaHAD breast cancer weights to impart foundational histopathological priors and baseline H&E staining features. Training was performed in a class-conditional setting on an NVIDIA GeForce RTX 5090 GPU, integrating horizontal flips as dataset-level augmentation while maintaining an ADA target probability threshold of 0.6 to ensure stable adversarial dynamics. Generative quality was tracked throughout 3,000 kimg, employing 50,000-sample Fréchet Inception Distance (FID-50k) evaluations to identify the lowest-distance checkpoint for downstream validation. From a transfer baseline of 131.19, the FID-50k declined to a minimum of 6.98 at the 2,800-kimg checkpoint.</p>

    <div class="figure">
        <img src="{fig1}" alt="FID Progression">
        <div class="caption"><span class="caption-bold">Fig. 1.</span> FID score progression during GAN training highlighting the global minimum of 6.98 at 2,800 kimg.</div>
    </div>

    <p>To quantify the trade-off between cellular-level realism and phenotypic variation, late-stage checkpoints were benchmarked against manifold-based Precision and Recall metrics.</p>

    <div class="figure">
        <img src="{fig2}" alt="Fidelity vs Diversity">
        <div class="caption"><span class="caption-bold">Fig. 2.</span> Checkpoint evaluation balancing image fidelity (Precision) against dataset diversity (Recall) across late-stage training snapshots.</div>
    </div>

    <p>Checkpoint 2,800 demonstrated the most favourable fidelity-diversity trade-off while preserving substantial biological diversity relative to earlier snapshots (Precision: 0.6825, Recall: 0.4332). Based on this optimal trade-off, the 2,800 kimg weights were preserved to generate a dedicated bank of 25,000 synthetic patches for all downstream stress tests and augmentation experiments.</p>

    <div class="figure">
        <img src="{fig3}" alt="Qualitative Comparison">
        <div class="caption"><span class="caption-bold">Fig. 3.</span> Qualitative comparison of synthetic patches generated by StyleGAN2-ADA against authentic clinical biopsies across the five ORCHID categories (Normal, OSMF, WDOSCC, MDOSCC, PDOSCC).</div>
    </div>

    <h2 class="sec-heading">IV. Downstream Classification Methodology</h2>
    <h3 class="subsec-heading">A. The Pure-Synthetic Stress Test</h3>
    <p>Standard generative metrics, most notably the Fréchet Inception Distance (FID), evaluate statistical visual fidelity, fundamentally decoupling perceptual plausibility from downstream clinical veracity. To evaluate whether the generated tiles retained authentic diagnostic utility, we implemented a pure-synthetic stress test using zero real images during training. Under this zero-real experimental protocol, downstream classifiers were trained exclusively on an artificially synthesized repository of 25,000 patches (5,000 samples per class across the five-tier benchmark) with zero exposure to authentic clinical tissue during optimization. The diagnostic generalizability of these zero-real representations was strictly evaluated on an independent, holdout test set composed entirely of real clinical patient biopsies. By isolating synthetic representations during training, this protocol ensures that classifier performance solely depends on the generator’s ability to encode genuine microscopic morphology.</p>

    <h3 class="subsec-heading">B. α Ablation Strategy</h3>
    <p>To identify the precise empirical threshold where synthetic data benefits or impairs classification, we conducted systematic mixing experiments combining synthetic and authentic cohorts. To calibrate synthetic augmentation, we defined a mixing parameter, α ∈ {0.0, 0.25, 0.50, 0.75, 1.0}, representing the proportion of the minority data deficit filled by generated patches. The volume of synthetic patches allocated to class c, denoted as S_c, is defined as:</p>

    <div class="equation">
        S<sub>c</sub> = [α × (T<sub>target</sub> - N<sub>c</sub>)]
        <span class="eq-num">(1)</span>
    </div>

    <p>where α denotes the fraction of the class deficit filled using generated images, N_c is the number of real images originally available in class c, and T_target is the required target volume in the combination of real and synthetic data. For example, if a class contains N_c = 1,500 real images and the target is T_target = 5,000, the deficit is 3,500 images. At α = 0.50, 1,750 synthetic images are added, producing a composite class size of 3,250.</p>
    <p>In the controlled balancing sweep (Experiment 3), T_target was fixed to the sample frequency of the primary majority class, standardizing the upper ceiling of each class prior to synthetic interpolation. On the other hand, in our aggressive dataset expansion sweep (Experiment 4), T_target was scaled to an artificial target of 5,000 images per class.</p>

    <h3 class="subsec-heading">C. Network Architectures and Preprocessing</h3>
    <p>To ensure our findings were independent of architectural bias, we benchmarked two contrasting architectures: DenseNet-121, which utilizes iterative feature concatenation, and EfficientNet-B3, which applies adaptive compound scaling. For both architectures, the native ImageNet classifier of each network was replaced with a custom head consisting of a dropout layer and a linear projection to the five ORCHID diagnostic classes.</p>
    <p>To address varying data configurations across phases, spatial resolutions, augmentations, and optimization objectives were systematically adjusted to match the analytical targets of each evaluation phase. To standardize training dynamics, regularization techniques were applied, including label smoothing (ε = 0.1), gradient clipping bounded by a maximum norm of 1.0, and PyTorch automatic mixed precision (AMP) to maintain robust numerical convergence. The distinct preprocessing routines and hyperparameter settings for each classification experiment are detailed in Table II.</p>

    <div class="table-container">
        <div class="table-title">TABLE II<br>Preprocessing and Hyperparameter Configurations across Experimental Phases</div>
        <table class="ieee-table">
            <thead>
                <tr>
                    <th style="text-align:left;">Experimental Phase</th>
                    <th>Exp 2: Pure Synthetic Stress Test</th>
                    <th>Exp 3: Controlled Balancing Sweeps</th>
                    <th>Exp 4: Aggressive Target Expansion</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Image Resolution & Augmentations</strong></td>
                    <td>300×300 px.<br>Random H/V flips, rotations up to 15°.</td>
                    <td>224×224 px.<br>RandomResizedCrop(0.8, 1.0), H/V flips, 15° rot, 15% color jitter.</td>
                    <td>300×300 px.<br>H/V flips, 15° rotations, standard color jittering.</td>
                </tr>
                <tr>
                    <td><strong>Optimization & Scheduler</strong></td>
                    <td>25 Epochs (AdamW).<br>OneCycleLR (Backbone: 10⁻⁴, Head: 10⁻³).</td>
                    <td>25 Epochs (AdamW).<br>CosineAnnealingLR (Base: 3×10⁻⁴, 3-epoch warmup).</td>
                    <td>20 Epochs (AdamW).<br>CosineAnnealingLR (Base: 10⁻³ tapering to 10⁻⁵).</td>
                </tr>
                <tr>
                    <td><strong>Dropout & Weight Decay</strong></td>
                    <td>Dropout: p = 0.4<br>WD: 10⁻²</td>
                    <td>Dropout: p = 0.3<br>WD: 10⁻⁴</td>
                    <td>Dropout: p = 0.3<br>WD: 10⁻⁴</td>
                </tr>
            </tbody>
        </table>
    </div>

    <h2 class="sec-heading">V. Experimental Results & Downstream Benchmarks</h2>
    <h3 class="subsec-heading">A. Generative Benchmarking and the Pure-Synthetic Generalization Gap</h3>
    <p>The conditional StyleGAN2-ADA framework demonstrated robust convergence, reaching an optimal FID-50k of 6.98 at the 2,800-kimg snapshot before reaching asymptotic stability. This checkpoint established an optimal trade-off between sample fidelity (Precision = 0.6825) against structural diversity (Recall = 0.4332). Nonetheless, the pure-synthetic stress test revealed a significant gap between superficial generative realism and biological ground truth. When trained solely on the 25,000-sample synthetic bank, both DenseNet-121 and EfficientNet-B3 rapidly exceeded 99% training accuracy with smooth loss decay.</p>

    <div class="figure">
        <img src="{fig4}" alt="Training Accuracy">
        <div class="caption"><span class="caption-bold">Fig. 4.</span> Synthetic optimization: Training accuracy trajectories of DenseNet-121 and EfficientNet-B3 under the pure-synthetic protocol.</div>
    </div>

    <div class="figure">
        <img src="{fig5}" alt="Training Loss">
        <div class="caption"><span class="caption-bold">Fig. 5.</span> Synthetic optimization: Training loss decay curves over 25 epochs.</div>
    </div>

    <p>When tested on authentic patient biopsies, both models suffered severe performance collapse, indicating that features learned from the synthetic manifold failed entirely to generalize to genuine pathological tissue specimens. As optimization advanced, validation loss diverged sharply upward, resulting in a severe generalization gap when tested against real patient biopsies.</p>

    <div class="figure">
        <img src="{fig6}" alt="Generalization Gap">
        <div class="caption"><span class="caption-bold">Fig. 6.</span> Generalization gap in DenseNet-121 and EfficientNet-B3 illustrating severe divergence between training accuracy on synthetic data and validation performance on genuine tissue.</div>
    </div>

    <div class="figure">
        <img src="{fig7}" alt="Clinical Collapse">
        <div class="caption"><span class="caption-bold">Fig. 7.</span> Pure-synthetic stress test: Real-world clinical collapse showing diagnostic accuracy dropping below 10% on real patient biopsies.</div>
    </div>

    <p>Real-world diagnostic performance collapsed drastically to 9.17% for DenseNet-121 and 9.36% for EfficientNet-B3, accompanied by single-digit macro F1-scores. This profound ~90% collapse highlights a fundamental limitation of standard generative metrics, i.e., these metrics fail to assess whether synthetic images preserve authentic clinical markers. While StyleGAN2-ADA captured macro-level H&E staining profiles and broad stromal textures, the model failed to encode the subtle cytological markers critical for reliable clinical grading.</p>

    <h3 class="subsec-heading">B. Bounded Regularization via Controlled Balancing</h3>
    <p>In the controlled balancing protocol, synthetic patches were introduced selectively to offset minority deficits, capping each class at the real majority volume and increasing the dataset from 10,228 to 13,950. Under this bounded regime, synthetic additions didn’t distort authentic biological variance and served as an effective regularizer for the downstream classifiers.</p>

    <div class="figure">
        <img src="{fig8}" alt="Bounded Balancing Accuracy">
        <div class="caption"><span class="caption-bold">Fig. 8.</span> Experiment 3: Test accuracy vs. mixing parameter α under bounded balancing sweeps.</div>
    </div>

    <div class="figure">
        <img src="{fig9}" alt="F1 Heatmap">
        <div class="caption"><span class="caption-bold">Fig. 9.</span> Experiment 3: Macro F1-score heatmap across α sweeps for DenseNet-121 and EfficientNet-B3.</div>
    </div>

    <p>The two backbones exhibited divergent performance trajectories, driven by the distinct structural mechanics of each architecture. DenseNet-121 was sensitive to synthetic noise due to its dense feature concatenation, achieving its performance ceiling on the purely authentic baseline (96.80% test accuracy). In contrast, EfficientNet-B3 leveraged its compound scaling and channel attenuation to suppress high-frequency generative artifacts while exploiting the underlying morphological variance of the synthesized patches. This lifted EfficientNet-B3’s performance from an authentic baseline test accuracy of 96.29% to a performance zenith of 96.42% at full minority class balance (α = 1.00). This demonstrates that when constrained to authentic class limits, synthetic augmentation effectively mitigates class imbalance without compromising diagnostic accuracy.</p>

    <h3 class="subsec-heading">C. The Synthetic Dilution Threshold</h3>
    <p>Experiment 4 evaluated the boundaries of generative regularization by expanding every diagnostic class to an artificial ceiling of 5,000 images per class. At maximum expansion (α = 1.00), the dataset swelled to 25,000 images, shifting the data distribution such that synthetic samples accounted for roughly 59% of the total volume. Such extensive saturation caused severe feature dilution, obscuring critical sub-cellular diagnostic features beneath repetitive, synthetic-manifold regularities. By overwhelming authentic tissue samples, generative artifacts corrupted the models’ discriminative decision boundaries and degraded grading accuracy.</p>

    <div class="figure">
        <img src="{fig10}" alt="Dilution Threshold">
        <div class="caption"><span class="caption-bold">Fig. 10.</span> Experiment 4: The synthetic dilution threshold showing performance degradation beyond critical synthetic saturation.</div>
    </div>

    <div class="figure">
        <img src="{fig11}" alt="Architectural Resilience">
        <div class="caption"><span class="caption-bold">Fig. 11.</span> Architectural resilience to generative dilution comparing DenseNet-121 and EfficientNet-B3.</div>
    </div>

    <p>The declining accuracy trajectories reflect this systematic degradation. DenseNet-121 suffered a substantial performance drop in test accuracy from 95.67% to 93.22%. EfficientNet-B3 exhibited greater robustness against generative noise propagation, but still suffered an absolute decline of 0.94%, falling from 96.98% to 96.04%.</p>

    <div class="figure">
        <img src="{fig12}" alt="Per-Class F1 Degradation">
        <div class="caption"><span class="caption-bold">Fig. 12.</span> Experiment 4: Per-class F1 degradation at maximum synthetic saturation across both backbones.</div>
    </div>

    <p>Granular class-level heatmaps demonstrate that dilution was widespread, systematically degrading precision and recall across all diagnostic categories. Ultimately, these results establish a strict operational boundary for clinical AI: synthetic histopathology effectively regularizes minority imbalances, but overwhelming authentic tissue distributions actively impairs diagnostic performance.</p>

    <h2 class="sec-heading">VI. Discussion: Pathological Failure Modes</h2>
    <p>To investigate the exact failure mechanics of the pure-synthetic regime, we examined the normalized confusion matrices and cross-class predictions from Experiment 2.</p>

    <div class="figure">
        <img src="{fig13}" alt="DenseNet-121 Confusion Matrix">
        <div class="caption"><span class="caption-bold">Fig. 13.</span> DenseNet-121: Normalized confusion matrix under pure-synthetic training, highlighting severe cross-class collapse into the Normal category.</div>
    </div>

    <div class="figure">
        <img src="{fig14}" alt="EfficientNet-B3 Confusion Matrix">
        <div class="caption"><span class="caption-bold">Fig. 14.</span> EfficientNet-B3: Normalized confusion matrix illustrating pervasive collapse of OSMF into normal tissue and PDOSCC into WDOSCC.</div>
    </div>

    <p>The most severe failure occurred in the Oral Submucous Fibrosis (OSMF) class across both deep backbones. DenseNet-121 and EfficientNet-B3 erroneously classified 86.6% and 94.2% of authentic biopsies into the normal category, thereby collapsing the decision margin between healthy tissue and premalignant mucosal transformation. While the generator mimicked the general eosinophilic appearance of connective tissue, it failed to resolve the dense, bundled collagen fibrillogenesis and progressive stromal hyalinization that pathognomonically demarcate OSMF from normal mucosal architecture. A secondary diagnostic breakdown emerged within the OSCC grading hierarchy. EfficientNet-B3 misclassified 74.9% of aggressive PDOSCC cases as WDOSCC, indicating a collapse in tumour-grade separation. Clinical grading is determined by precise cellular features, especially nuclear-to-cytoplasmic ratios and the identification of distinct keratin pearls. The generated samples replaced discrete cellular phenotypes with smoothed malignant textures, leaving the classifiers unable to detect genuine diagnostic markers.</p>

    <h2 class="sec-heading">VII. Conclusion</h2>
    <p>This research reveals a critical vulnerability dissociation in generative medical AI: visual realism is clinically deceptive and masks the loss of key diagnostic features. Although StyleGAN2-ADA achieved remarkable qualitative fidelity and a low Fréchet Inception Distance (FID = 6.98), the synthetic data failed pure-synthetic stress tests, lacking the microscopic biological variance needed for sustaining independent clinical-grade classification. However, synthetic data remains highly valuable when strictly constrained: our empirical mixing experiments demonstrate that bounded augmentation functions as an effective regularizer to resolve minority-class imbalances. In contrast, aggressive synthetic over-saturation dilutes genuine clinical signals, compromising the classifiers’ discriminative margins. In the final analysis, high generative fidelity cannot be conflated with clinical validity. Future research must look beyond natural-image FID scores and necessitate downstream clinical stress tests before deploying synthetic data prior to clinical deployment.</p>

    <h2 class="sec-heading">References</h2>
    <div class="ref-item">[1] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in <em>Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)</em>, 2017, pp. 4700-4708.</div>
    <div class="ref-item">[2] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in <em>Int. Conf. Mach. Learn. (ICML)</em>, 2019, pp. 6105-6114.</div>
    <div class="ref-item">[3] L. Oakden-Rayner et al., "Hidden stratification causes clinically meaningful failures in machine learning for medical imaging," in <em>Proc. ACM Conf. Health, Inference, and Learning</em>, 2020, pp. 151-159.</div>
    <div class="ref-item">[4] R. Chaudhary et al., "ORCHID: A multi-center open-access oral cancer histology image database," <em>Sci. Data</em>, vol. 11, p. 123, 2024.</div>
    <div class="ref-item">[5] S. Chakrabarty et al., "YOLO26/CNN benchmarking on the ORCHID oral cancer dataset," <em>Preprint</em>, 2026.</div>
    <div class="ref-item">[6] R. Chaudhary et al., "OralPatho framework on whole slide images," <em>medRxiv</em>, 2023.</div>
    <div class="ref-item">[7] A. Kumar, B. Kumar, and C. Jindal, "EfficientNet-B0 on balanced 3k ORCHID subset," <em>IEEE Trans. Med. Imaging</em>, 2026.</div>
    <div class="ref-item">[8] Y. Ren, X. Li, and Z. Chen, "Computer-aided classification study for OSCC," <em>Front. Oncol.</em>, vol. 15, 2026.</div>
    <div class="ref-item">[9] A. Tasnim et al., "Histo-AdaptiveViT: Long-tailed classification for histopathology," in <em>Proc. IEEE Int. Symp. Biomed. Imaging (ISBI)</em>, 2026.</div>
    <div class="ref-item">[10] I. Goodfellow et al., "Generative adversarial nets," <em>Adv. Neural Inf. Process. Syst. (NeurIPS)</em>, vol. 27, 2014.</div>
    <div class="ref-item">[11] T. Karras et al., "Training generative adversarial networks with limited data," <em>Adv. Neural Inf. Process. Syst. (NeurIPS)</em>, vol. 33, pp. 12104-12114, 2020.</div>
    <div class="ref-item">[12] J. Dee et al., "Domain adaptation with StyleGAN2 on thyroid histopathology," <em>PLOS ONE</em>, vol. 19, 2024.</div>
    <div class="ref-item">[13] M. Heusel et al., "GANs trained by a two time-scale update rule converge to a local Nash equilibrium," <em>Adv. Neural Inf. Process. Syst.</em>, vol. 30, 2017.</div>
    <div class="ref-item">[14] M. Chong and D. Forsyth, "Effectively unbiased FID and inception score and where to find them," in <em>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</em>, 2020, pp. 6070-6079.</div>
    <div class="ref-item">[15] Y. Xue et al., "Selective synthetic augmentation with HistoGAN for breast cancer pathology," <em>Med. Image Anal.</em>, vol. 67, 2021.</div>
    <div class="ref-item">[16] M. Frid-Adar et al., "GAN-based synthetic medical image augmentation for increased CNN performance in liver lesion classification," <em>Neurocomputing</em>, vol. 321, pp. 321-331, 2018.</div>

</div>

</body>
</html>
"""

html_path = os.path.join(BASE_DIR, 'IEEE_Paper.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"HTML saved to: {html_path}")

pdf_path = os.path.join(BASE_DIR, 'IEEE_Paper.pdf')
chrome_cmd = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "--headless",
    "--disable-gpu",
    f"--print-to-pdf={pdf_path}",
    "--no-pdf-header-footer",
    html_path
]
res = subprocess.run(chrome_cmd, capture_output=True, text=True)
if os.path.exists(pdf_path):
    print(f"PDF successfully generated at: {pdf_path} (size: {os.path.getsize(pdf_path)} bytes)")
else:
    print(f"Chrome error: {res.stderr}")
