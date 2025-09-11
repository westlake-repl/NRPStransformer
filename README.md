[![Website Status Check](https://github.com/westlake-repl/NRPSTransformer/actions/workflows/http_check.yml/badge.svg)](https://github.com/westlake-repl/NRPSTransformer/actions/workflows/http_check.yml)
[![DOI](https://zenodo.org/badge/961756607.svg)](https://doi.org/10.5281/zenodo.15773640)

# NRPSTransformer
**A Transformer-Based Predictor for Nonribosomal Peptide Synthetases (NRPS) Specificity**

Welcome! You can try our web demo here: http://www.nrpstransformer.cn

## Overview
NRPSTransformer is a deep learning model designed to predict the substrate specificity of adenylation (A) domains in Nonribosomal Peptide Synthetases (NRPS). It employs a transformer-based architecture that capitalizes on transfer learning. It generates high-quality protein embeddings from a large, pre-trained language model, which are subsequently fed into a fine-tuned classifier to achieve state-of-the-art prediction accuracy of 93%.

## Publication
This work has been published in the Journal of the American Chemical Society (JACS). 

**DOI:** [10.1021/jacs.5c08076](https://doi.org/10.1021/jacs.5c08076)

## Prerequisites
Before you begin, ensure you have the following installed:

- **Conda/Miniconda**: To manage the project environment. You can find installation instructions here.
- **Git LFS**: For handling large model files. You can install it by following the instructions here.
- **HMMER**: For domain identification and scanning. Install it using:
  ```bash
  # On Ubuntu/Debian
  sudo apt-get install hmmer
  
  # On macOS with Homebrew
  brew install hmmer
  
  # On CentOS/RHEL
  sudo yum install hmmer
  ```

## Installation
Please follow these steps carefully to set up the project environment and download the necessary models.

**1. Clone the Repository**
First, clone this repository to your local machine.

 ```bash
git clone https://github.com/westlake-repl/NRPSTransformer.git
cd NRPSTransformer
 ```

**2. Set Up the Conda Environment**
Create and activate a dedicated conda environment using the provided requirement.txt file.

```bash
# Create a new conda environment named 'NRPS' with Python 3.10
conda create -n NRPS python=3.10 -y
# Activate the newly created environment
conda activate NRPS
# Install the required Python packages
pip install -r requirements.txt
```

**3. Download the ESM-2 Backbone Model**
This project uses the esm2_t33_650M_UR50D model as a backbone. You need Git LFS to clone it from Hugging Face.
```bash
# Initialize Git LFS
git lfs install
# Clone the ESM-2 model repository
git clone https://huggingface.co/facebook/esm2_t33_650M_UR50D
```

**4. Download Project Checkpoints and Labels**
The model checkpoints and class labels are hosted on Zenodo. Download them and place them in the correct directories.
```bash
# Create the necessary directories
mkdir -p checkpoints
mkdir -p model/class_label

# Download checkpoint files into the 'checkpoints' directory
wget -O checkpoints/all.ckpt https://zenodo.org/records/15771488/files/all.ckpt
wget -O checkpoints/benchmark.ckpt https://zenodo.org/records/15771488/files/benchmark.ckpt
wget -O checkpoints/clade.ckpt https://zenodo.org/records/15771488/files/clade.ckpt

# Download label mapping files into the 'model/class_label' directory
wget -O model/class_label/labelid2label-17.pt https://zenodo.org/records/15771488/files/labelid2label-17.pt
wget -O model/class_label/labelid2label-43.pt https://zenodo.org/records/15771488/files/labelid2label-43.pt
```

After completing these steps, your project directory should look like this:

```bash
NRPSTransformer/
├── checkpoints/
│   ├── all.ckpt
│   ├── benchmark.ckpt
│   └── clade.ckpt

├── model/
│   ├── class_label/
│   │   ├── labelid2label-17.pt
│   │   └── labelid2label-43.pt
│   ├── esm2_t33_650M_UR50D/
│   └── ... (Hugging Face model files)
├── requirements.txt
├── run.sh
└── ... (other project files)
```

## Usage
The primary way to run predictions is through the run.sh script. It provides an interactive command-line interface to specify your input and output files.

**1. Execute the Script**
Make sure you are in the NRPSTransformer root directory and the NRPS conda environment is active.
```bash
bash run.sh
```

**2. Interactive Prompts**
The script will prompt you for two pieces of information:

- **Input File Path**: The path to your FASTA file containing the protein sequences of the A-domains you want to predict.
- **Result File Path**: The path where the prediction results will be saved in CSV format.
You can either type a new path or press Enter to accept the default values shown in the parentheses.

Here is an example of the terminal interaction:

```bash
Please enter the input sequence file path (Default: ./sequence/sequence.fasta):
# User can type a path like './my_sequences.fasta' or just press Enter

setting input file to ./sequence/sequence.fasta

Please enter the result file path (Default: ./result/result.csv): 
# User can type a path like './my_results.csv' or just press Enter

setting result path to ./result/result.csv

# The prediction process will start now...
# Running HMM scan...
# Parsing domain results...
# Generating domain sequences...
# Running AI prediction...
# Time: 45.2sec
# Successfully finished all steps!
# The result is saved to ./result/result.csv
```

**Example Input File Format:**
Your input FASTA file should contain protein sequences with A-domains, like this:

```fasta
>pepstatin-pepB(leu)
MSADTALSSAQQRLWFLWQLRPDGNEYNVPRATRLRGALDSDALRRALDLLVARHDALRATFPTVAGAPVLRIAAEAEVPLPLTDLSALPTAERDAALDEAVTRAALAPFDLAHGPVLRAELIRLAEDDHALVLTFHHIAVDGWSMGIIDRDLAALYAATVAGDDAPSAAPARSYRDCVAAEQRLLAGPRRHEALDHWRTELAGAPLELALPTDRPHPTAPSFAGDSRGFPVPPELAERLDAVAVRHRVTRFIALLSGYAALLSRMTSTPDLVIGVPVSGRTEMEVEGTVGLFVNMMPVRIRCTADTTFGALLHQVRDTVLAGHEYQDLPFQLLVEEVQPDRSTARHPLFQTVVTYEDLPTEGTALPGLDASPLPIPTRTAKYELALHLAGNRQRTEAWVGYQTDVFDDRSAELIGDRYLRLLSAALDDPDTPVAQLPVLGSEEERLLLTDWAGAPERAAPCPARERVDQLVERQARAHPDAVAVRSDEGALSYAALDRAADRIAAGLRAAGAGPGTVVATCLPRGADLVTAQLGVLKSGAAYLSLDPAHPAPRLTALLAEARPLLTLTGPEHRAGLEYGRVETLHSLRDAGDAPDAPDVPDAPDGPPAASAGPDDLAYVMYTSGSTGRPKGVLVEHHALANLVAWHRAEFGIGPGDRCTLVAAPGFDASVWETWSALTAGATVEVPSAETVLSPSELRSWLIERGITSAFLPTPLLERMLQEPWPAASALRSVLTGGDRLHGTGRNPLPFRLVNNYGPTENTVVATSGTVPADGGERDTLPGIGRPVTGTEAYVLDAELRPVPVGVPGELYLGGEGLARGYLGQPALTADRFVPHPFSRTPGARLYRTGDLVRWRTDAGLDFLGRNDHQVKIRGIRVEPGEIENVLRAHPGVHDAVVAAGSPDGTAEPELTAYLVPADASGAPDLAALLDHTTRRLPRHMHPRRYLLLPSVPLTANGKVDRTALAGAARELAPPPAAAHRRARTPLERLISDVWALTLGHDSFGTDDNFFDAGGHSLLLASVRDRLTAELGTPVRIADLYAYPTIATLARQLSGSPGSTAEERPAGRTGGDGVAERRRRGAARLTAMRARTSHGRAERQ

>pepstatin-pepD(ala)
MTDLTATAPGPTGTADADPSHPASFAQRRLWFLDQLDSGNAAYNLMAALRLRGPLDTEALHWSLNRIVDRHEVLRTVFRADGEGEPRQRVTPHRPLDLPVTDLTGQPREERDERARTLIETETERAFDLATGPLIRTRVVRLDQREHVLAVICHHTVCDGWSMDRLFAELSELYAARTAGRDPALEPLPLQFGEYAARQRGELAGAEARAALAHWRTRLQGAPTLLELPTSFPRPAVQGFDGATHTVPLSPGTWQEVRAAAARHKATPFMLLLSVFALQLSRLSGADDLVIGSPSAGRGRPELAPLIGMFVNTLPLRIDLSGEVTFEELLRRVRRSALDAFPRQDVPLERIVTELGPERARSHDPLFQTMFALQQPLSAPELAGLATDIFPASPRTTFTDLWLEIRPHAAGSDGTGGADCCFRYRTELFDAETVGRLARQYQHLLRTALDAPGTRLSRFSLTGEQETARLLRQGNRSAQAHPWDGPVHEAVDRQARRTPGATAVVFADERLSYAQLAGRTGQLARHLAAHGAAADTPVAVCLPRGTGLVTSVLAVMASGSAYLPLSPEDPPARLVRMLRAAGAAHILTTRELAPVLADAGVPVTAVDTFPWEEGGWPDRPMAPGRVGPDHLAYVIHTSGSTGAPKAVGVPHRGLANRIRGIQDTHGLDTTDRVLHKTPVTFDVSLWELLWPLTVGATLVVAAPGGHRDPTYLVELIERERVTTVHFVPPLLAAFLEEPGLHRCASLRRVLCSGQELPRATRDRCLERLPARLFNAYGPTEASIEVTEEECAPKATGGGAPDPRVSIGRPIAGAEVYVLDAELRPVPAGVPGELYLGGVALARGYLGQPALTADRFVPHPFSRTPGARLYRTGDLVRWRSDAALDFLGRNDHQVKIRGIRVEPGEIENVLRAHPDVRDAVVVATRPDGATGIQLTAYLVPYATDGSRAAPDADALVAAVDDRLRASLPGFMVPSRTVVLPRLPLNASGKVDRAALPAPAAPAPAPATARVAPADHVESTLARIWSQVLDRPGTDVTEDFFALGGDSLKSIQLVHRAREAGLSLRVGDIFHHPTVRDLAAHVRRTAAPAEPREDR
```

The program will then run the prediction pipeline, and upon completion, a message will confirm that the results have been saved to your specified output file.

## Workflow Overview
NRPSTransformer follows a comprehensive pipeline to predict NRPS substrate specificity:

**1. HMM Domain Scanning**
- Uses HMMER to scan input protein sequences for A-domain regions
- Identifies conserved motifs and domain boundaries
- Generates domain-specific sequences for further analysis

**2. Domain Extraction and Processing**
- Parses HMM scan results to extract A-domain sequences
- Validates domain quality and filters out low-confidence predictions
- Prepares sequences in the format required by the transformer model

**3. AI-Based Prediction**
- Loads the pre-trained ESM-2 transformer model
- Applies fine-tuned classification layers for substrate specificity
- Generates confidence scores for multiple substrate predictions

**4. Result Compilation**
- Combines predictions with original sequence information
- Provides top-3 substrate predictions with confidence scores
- Outputs results in CSV format for easy analysis

## Output Format
The prediction results are saved as a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `ID` | Original sequence identifier from input FASTA |
| `Domain` | Extracted A-domain sequence |
| `Top-1(score)` | Highest confidence substrate prediction with score |
| `Top-2(score)` | Second highest confidence substrate prediction with score |
| `Top-3(score)` | Third highest confidence substrate prediction with score |

**Example Output:**
```csv
ID,Domain,Top-1(score),Top-2(score),Top-3(score)
pepstatin-pepB(leu),LAYVMYTSGSTGRPKGVLVEHHALANLVAWHRAEFGIGPGDRCTLVAAPGFDASVWETWSALTAGATVEVPSAETVLSPSELRSWLIERGITSAFLPTPLLERMLQEPWPAASALRSVLTGGDRLHGTGRNPLPFRLVNNYGPTENTVVATSGTVPADGGERDTLPGIGRPVTGTEAYVLD,gln(0.9978),ser(0.0011),hty(0.000151)
pepstatin-pepD(ala),LAYVIHTSGSTGAPKAVGVPHRGLANRIRGIQDTHGLDTTDRVLHKTPVTFDVSLWELLWPLTVGATLVVAAPGGHRDPTYLVELIERERVTTVHFVPPLLAAFLEEPGLHRCASLRRVLCSGQELPRATRDRCLERLPARLFNAYGPTEASIEVTEEECAPKATGGGAPDPRVSIGRPIAGAEVYVLD,asp(0.5783),asn(0.3875),salicylate(0.0146)
pepstatin-pepG-SA(val),PAYVIYTSGSSGDPKGVEVSHRNVTALLAACDRVFALRGDDVWTLFHSPCFDFSVWEMWGALAHGAKLVVVPAEVARSPEATLDLVVSEGVTVLNQVPSVFRYLSRSAVARAEDTADRPATALRYVIFGGEPVDVDAVRAWRALHGTRTEFFNMYGITETTVFATCRRLPESEIDPRPTTAPDAPAPTGTTADPELNIGRPLDGFEVVLLD,tyr(0.9705),ser(0.007952),cys(0.006917)
syrilipamide-nrps(leu-ser-hse-val-bala-ser),LAYVIYTSGSTGQPKGVMIEHRNLVNLVAWHCEAFGLTHRKRVSSVAGVGFDACVWELWPALCVGASLSLLPGQALGNDVDALLGWWRRQDLDVSFLPTPIAEIAFAQGIEPASLQTLLIGGDRLRQFPNPDSRVALINNYGPTETTVVATSGLIDATQSVLHIGRPIANTQVYLLD,gln(0.9996),arg(8.013e-05),ser(6.634e-05)
```

## Advanced Usage

### Direct Python Script Execution
For users who prefer more control over the prediction process, you can run individual components directly:

**1. HMM Domain Scanning**
```bash
# Run HMM scan on your sequences
./hmm/scan.sh your_sequences.fasta

# Parse the HMM results
python ./hmm/parse_dbtl.py

# Generate domain sequences
python ./hmm/gen_domains.py --fasta_path your_sequences.fasta
```

**2. Direct AI Prediction**
```bash
# Run prediction with custom parameters
python inference.py --inference_dataset hmm/result/domains.csv --result_path your_results.csv
```

### Custom Model Checkpoints
You can use different pre-trained model checkpoints for specific use cases:

- `all.ckpt`: General-purpose model trained on all available data (recommended)
- `benchmark.ckpt`: Model optimized for benchmark datasets
- `clade.ckpt`: Model specialized for specific bacterial clades

To use a different checkpoint, modify the `CHECKPOINT_PATH` variable in `inference.py`:
```python
CHECKPOINT_PATH = "checkpoints/benchmark.ckpt"  # or clade.ckpt
```

### Batch Processing
For processing multiple FASTA files, you can create a simple batch script:

```bash
#!/bin/bash
# batch_process.sh

for fasta_file in /path/to/your/fasta/files/*.fasta; do
    echo "Processing $fasta_file"
    python inference.py --inference_dataset "$fasta_file" --result_path "${fasta_file%.fasta}_results.csv"
done
```

## Troubleshooting

### Common Issues and Solutions

**1. HMMER Not Found**
```
Error: The input file 'sequence.fasta' does not exist.
```
**Solution:** Ensure HMMER is properly installed and accessible in your PATH:
```bash
which hmmsearch  # Should return the path to hmmsearch
hmmsearch -h     # Should show help information
```

**2. CUDA/GPU Issues**
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce batch size in `inference.py` (change `BATCH_SIZE = 8` to `BATCH_SIZE = 4` or `BATCH_SIZE = 2`)
- Use CPU-only mode by modifying the trainer configuration:
```python
trainer = pl.Trainer(
    accelerator="cpu",  # Change from "gpu" to "cpu"
    devices=1,
    logger=False,
    enable_progress_bar=False,
    enable_model_summary=False,
)
```

**3. Model Download Issues**
```
Error: Failed to download model files
```
**Solutions:**
- Ensure Git LFS is properly installed: `git lfs install`
- Check internet connection and try downloading manually:
```bash
wget -O checkpoints/all.ckpt https://zenodo.org/records/15771488/files/all.ckpt
```

**4. Memory Issues**
```
MemoryError: Unable to allocate array
```
**Solutions:**
- Close other applications to free up RAM
- Process sequences in smaller batches
- Use a machine with more available memory

**5. No A-domains Found**
```
Warning: No A-domains detected in input sequences
```
**Solutions:**
- Verify your input sequences contain NRPS A-domains
- Check that sequences are in proper FASTA format
- Ensure sequences are long enough (>200 amino acids)
- Try with known NRPS sequences first

**6. Permission Issues**
```
Permission denied: cannot execute script
```
**Solutions:**
```bash
chmod +x run.sh
chmod +x hmm/scan.sh
```

### Getting Help
If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/westlake-repl/NRPSTransformer/issues) page
2. Verify your environment matches the requirements
3. Try running with the provided example sequences first
4. Check the log files in the project directory for detailed error messages

## Citation
If you use NRPSTransformer in your research, please cite our paper:

```bibtex
@article{nrpstransformer2024,
  title={NRPSTransformer: A Transformer-Based Predictor for Nonribosomal Peptide Synthetases Specificity},
  author={[Authors]},
  journal={Journal of the American Chemical Society},
  year={2024},
  doi={10.1021/jacs.5c08076}
}
```

## Contributing
We welcome contributions to NRPSTransformer! Here's how you can help:

### Reporting Issues
- Use the [GitHub Issues](https://github.com/westlake-repl/NRPSTransformer/issues) page to report bugs
- Include detailed information about your environment and the error
- Provide example sequences that reproduce the issue (if applicable)

### Contributing Code
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -m "Add feature"`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request
