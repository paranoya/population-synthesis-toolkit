# Ingestion of Bruzual & Charlote 2003 models

- [Link](http://www.bruzual.org/bc2003_models/) to the website

Choose a model version and download the files and the src code

Follow the instructions in section 3 of the [original documentation](https://www.bruzual.org/bc03/doc/bc03.pdf) to compile GALAXEV.

Run the bash script `all_ised2fits.sh` provided in this directory providing the path to the parent directory of the *ised files:

```
bash all_ised2fits.sh /SSP_TEMPLATES/BC03/bc03_2013ver/bc03/Padova1994/chabrier
```

In the same directory you should find the new `.fits` files containing the SED of each SSP.

