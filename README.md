# BaySer

BaySer is a Python package for Bayesian seriation of archaeological assemblages, with optional radiocarbon-informed calendar modelling.

It estimates latent assemblage positions from a binary assemblage-by-type matrix, compares the inferred order with a classical reciprocal-averaging/correspondence-analysis baseline, and can link dated assemblages to calendar time through an IntCal20 radiocarbon likelihood. Optional outlier components can be used to inspect cases where radiocarbon determinations and typological expectations are in tension.

BaySer is currently a research prototype. It is intended for transparent, reproducible and exploratory chronological analysis, not as a black-box replacement for archaeological judgement.

## Current scope

BaySer currently supports:

* binary presence/absence assemblage-by-type matrices
* one-dimensional latent Bayesian seriation
* optional assemblage-level richness effects
* comparison with a classical CA/RA baseline
* optional radiocarbon-informed calendar linkage using IntCal20
* optional date-level outlier components
* posterior rank summaries and order diagnostics
* CSV/JSON result export
* diagnostic plots for seriation, radiocarbon linkage and model comparison

Important current limitations:

* the model assumes a single dominant seriation axis
* input data are currently binary presence/absence matrices
* calendar linkage uses a simple monotonic linear mapping from latent position to calendar age
* outlier probabilities are diagnostics of model tension, not archaeological explanations
* the public API is still unstable and may change before a full release

## Installation

Clone the repository and install the project with `uv`:

```bash
git clone https://github.com/MartinHinz/bayser.git
cd bayser
uv sync
```

Run the command-line interface:

```bash
uv run bayser --help
```

BaySer currently requires Python 3.14 and uses PyMC for Bayesian inference.

## Minimal typology-only example

The Münsingen example can be run without radiocarbon dates:

```bash
uv run bayser \
  --features examples/munsingen/munsingen.csv \
  --results-dir outputs/munsingen/results \
  --plot-dir outputs/munsingen/plots \
  --draws 500 \
  --tune 500 \
  --chains 2 \
  --quiet
```

This writes result tables to `outputs/munsingen/results/` and diagnostic plots to `outputs/munsingen/plots/`.

## Radiocarbon-linked example

The Bronze Age example combines a binary feature matrix with radiocarbon determinations:

```bash
uv run bayser \
  --features examples/bronze_age_graves/feature_matrix.csv \
  --c14 examples/bronze_age_graves/14c.csv \
  --intcal20 data/intcal20.14c \
  --feature-id-col grave_id \
  --c14-id-col grave_id \
  --bp-col bp \
  --error-col error \
  --results-dir outputs/bronze_age/results \
  --plot-dir outputs/bronze_age/plots \
  --draws 500 \
  --tune 500 \
  --chains 2 \
  --quiet
```

A selected radiocarbon determination can be evaluated with an explicit outlier prior:

```bash
uv run bayser \
  --features examples/bronze_age_graves/feature_matrix.csv \
  --c14 examples/bronze_age_graves/14c.csv \
  --intcal20 data/intcal20.14c \
  --feature-id-col grave_id \
  --c14-id-col grave_id \
  --bp-col bp \
  --error-col error \
  --outlier ASO_6:0.5 \
  --results-dir outputs/bronze_age_ASO6/results \
  --plot-dir outputs/bronze_age_ASO6/plots \
  --draws 500 \
  --tune 500 \
  --chains 2 \
  --quiet
```

A low prior outlier probability can also be assigned to all retained dated assemblages:

```bash
uv run bayser \
  --features examples/bronze_age_graves/feature_matrix.csv \
  --c14 examples/bronze_age_graves/14c.csv \
  --intcal20 data/intcal20.14c \
  --feature-id-col grave_id \
  --c14-id-col grave_id \
  --bp-col bp \
  --error-col error \
  --outlier-all 0.05 \
  --results-dir outputs/bronze_age_global_outlier/results \
  --plot-dir outputs/bronze_age_global_outlier/plots \
  --draws 500 \
  --tune 500 \
  --chains 2 \
  --quiet
```

## Input formats

### Feature matrix

The main input is a CSV file with one row per assemblage and one column per artefact type. Values should be binary:

* `1` = type present in assemblage
* `0` = type absent from assemblage

An assemblage ID can either be supplied as the first column or specified explicitly with `--feature-id-col`.

Example structure:

```text
grave_id,type_a,type_b,type_c
G1,1,0,1
G2,0,1,1
G3,1,1,0
```

### Radiocarbon table

Radiocarbon-linked runs require a second CSV file with one row per dated assemblage. The assemblage ID column must match the feature matrix. The radiocarbon age and standard error columns are specified through CLI arguments.

Example structure:

```text
grave_id,bp,error
G1,3720,35
G2,3660,30
```

### Calibration curve

BaySer expects an IntCal20-format calibration curve file for radiocarbon-linked runs. The example data use `data/intcal20.14c`.

## Main outputs

BaySer writes outputs to the directory specified by `--results-dir`. Important files include:

* `metadata.csv` — run settings and high-level diagnostics
* `grave_summary.csv` — posterior assemblage positions, ranks and calendar summaries
* `type_summary.csv` — posterior type parameters and type ranks
* `score_comparison.csv` — comparison with the CA/RA baseline
* `chain_diagnostics.csv` — chain-wise order diagnostics
* `parameter_diagnostics.csv` — selected sampler diagnostics
* `posthoc_outlier_candidates.csv` — diagnostic screen for possible typology–radiocarbon tension
* `active_outliers.csv` — posterior outlier summaries where an outlier model was enabled
* `unmodelled_calibration.csv` — independent single-date calibration summaries for dated assemblages

Diagnostic plots are written to the directory specified by `--plot-dir`.

## Reproducibility notes

BaySer uses Bayesian MCMC sampling through PyMC. Results may vary slightly across package versions, operating systems and sampler settings. For reproducible analyses, record:

* BaySer version or Git commit
* Python version
* PyMC, PyTensor, NumPy, SciPy and ArviZ versions
* input data
* command-line arguments
* random seed
* generated `metadata.csv`

For the current web-demo release, dependencies should be pinned rather than left open-ended.

## Example data and sources

### IntCal20 calibration curve

The calibration curve included in `data/intcal20.14c` is IntCal20. Please cite the IntCal20 publication when using radiocarbon-linked BaySer runs:

Reimer, P. J., Austin, W. E. N., Bard, E., Bayliss, A., Blackwell, P. G., Bronk Ramsey, C., Butzin, M., Cheng, H., Edwards, R. L., Friedrich, M., Grootes, P. M., Guilderson, T. P., Hajdas, I., Heaton, T. J., Hogg, A. G., Hughen, K. A., Kromer, B., Manning, S. W., Muscheler, R., Palmer, J. G., Pearson, C., van der Plicht, J., Reimer, R. W., Richards, D. A., Scott, E. M., Southon, J. R., Turney, C. S. M., Wacker, L., Adolphi, F., Büntgen, U., Capano, M., Fahrni, S. M., Fogtmann-Schulz, A., Friedrich, R., Köhler, P., Kudsk, S., Miyake, F., Olsen, J., Reinig, F., Sakamoto, M., Sookdeo, A., & Talamo, S. 2020. The IntCal20 Northern Hemisphere Radiocarbon Age Calibration Curve (0–55 cal kBP). *Radiocarbon* 62(4), 725–757.

### Münsingen example

The Münsingen example is based on the classic Münsingen-Rain La Tène cemetery dataset, which has been used repeatedly as a benchmark for archaeological seriation and relative chronology.

Relevant sources include:

Hodson, F. R. 1968. *The La Tène Cemetery at Münsingen-Rain: Catalogue and Relative Chronology*. Bern: Stämpfli.

Kendall, D. G. 1971. Seriation from abundance matrices. In F. R. Hodson, D. G. Kendall & P. Tăutu (eds), *Mathematics in the Archaeological and Historical Sciences*, 215–252. Edinburgh: Edinburgh University Press.

### Bronze Age graves example

The Bronze Age graves example is derived from the dataset used by Brunner and colleagues for the chronology of the Central European Early Bronze Age and the transition to the Middle Bronze Age. The example is intended to demonstrate radiocarbon-linked Bayesian seriation and typology–radiocarbon tension diagnostics.

Please cite:

Brunner, M., von Felten, J., Hinz, M., Hafner, A., & Hübner, E. 2020. Central European Early Bronze Age chronology revisited: A Bayesian examination of radiocarbon dates and archaeological typology. *PLOS ONE* 15(12), e0243719.

## Development status

BaySer is under active development. The command-line interface and result file names are the most stable parts of the current workflow. Internal model functions, diagnostics and plotting utilities may change between releases.

## Citation

If you use Bayser, please cite the software using the metadata provided in `CITATION.cff`.

GitHub will display the recommended citation automatically when the repository is public. The citation information can also be exported from the repository page via **Cite this repository**.

If your work uses the example datasets or the IntCal20 calibration curve, please also cite the relevant original sources listed below.

## License

Bayser source code is released under the MIT License.

The example datasets and calibration data are included for demonstration and reproducibility. They should be cited according to their original sources and may be subject to their own reuse conditions.