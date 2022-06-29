# Ripple-Counting
## Introduction
Repository storing all final year project information, research, design and source code

## Project Structure

- 0_Ideation
- 1_Research
- 2_Rig
- 3_Software
- 4_Interim_Report
- 5_Final_Report

## Report Generation
### Requirements
```
sudo apt install pandoc

sudo apt install texlive-latex-recommended
```

### Generation
Once installed, the following command can be run to generate a pdf report.

*Ensure you are in the correct directory prior to executing the command*

```
pandoc report.md --bibliography references.bib --csl elsevier-harvard.csl --highlight-style mystyle.theme -o report.pdf
```