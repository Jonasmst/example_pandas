import sys
import os
import logging
try:
    import pandas as pd
    pd.options.mode.chained_assignment = None  # Suppress that pesky copy warning
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
except ModuleNotFoundError as error:
    print("ERROR: This script requires the following python libraries:")
    print("        - pandas (install using 'pip install pandas')")
    print("        - matplotlib (install using 'pip install matplotlib')")
    print("        - seaborn (install using 'pip install seaborn')")
    print("        - numpy (install using 'pip install numpy')")
    print("\nMissing package: %s" % error.name)
    sys.exit(-1)


def generate_dummy_data(num_genes=15804, num_samples=200):
    """
    Generates random data for use in dummy analyses. Generates a gene counts
    matrix, a gene annotation data set, and a sample annotation data set.

    Parameters
    ----------
    num_genes : int
        Number of genes in the generated data.
    num_samples: int
        Number of samples present in the generated data.

    Returns
    -------
    pandas.DataFrame, pandas.DataFrame
        One wide-format dataframe representing the gene counts matrix,
        one long-format dataframe representing the gene annotation data, and
        one long-format dataframe representing the sample annotation data.

    """
    logging.info("Generating dummy data for %d genes and %d samples" % (num_genes, num_samples))

    def generate_list_of_strings(strings, percentage_of_each_string, length):
        """
        Inner function used to create list of strings that follows a specified distribution. Used
        to generate e.g. gene types (50% protein-coding, 20% lncRNAs and 30% pseudogenes), and to
        generate dummy condition for sample annotations, e.g. 30% normal samples 70% cancer samples.

        Parameters
        ----------
        strings : list(str)
            List of the different strings to be present in the output.
        percentage_of_each_string: list(float)
            List of percentages, representing the desired distribution of the provided strings. Length and indexes must match "strings".
        length: int
            Number of values in total.
        
        Returns
        -------
        list(string)
            Distribution of strings, represented as a list.
        """
        # Calculate the number of occurrences of each string
        num_occurrences = np.array(percentage_of_each_string) / 100.0 * length
        num_occurrences = np.round(num_occurrences).astype(int)
        # Make sure that the number of occurrences adds up to the specified length
        num_occurrences[-1] = length - np.sum(num_occurrences[:-1])
        # Generate the list by repeating each string according to the number of occurrences
        result = []
        for i in range(len(strings)):
            result.extend([strings[i]] * num_occurrences[i])
        #print(pd.DataFrame({"whatever": result}).whatever.value_counts(normalize=True)*100)  # Verify distribution
        return result

    # Generate random gene counts matrix. Counts matrices are usually provided in wide-format, 
    # with gene identifier as row names and samples as columns. Each value represents the
    # the counts for a gene in a sample. E.g.:
    # gene_symbol   sample1     sample2     sample..    sampleN
    #   gene1       101         201         301         1001
    #   gene2       102         202         302         1002
    #   ...         ...         ...         ...         ...
    #   geneN       110         210         310         1010
    logging.info("\tPreparing gene counts matrix..")
    gene_counts_matrix = np.random.randint(low=0, high=10000, size=(num_genes, num_samples))  # +1 is for the column row when converting to a pandas DataFrame
    # Add a column with gene names to the counts matrix
    gene_names = ["gene_%d" % g for g in range(num_genes)]
    gene_counts_matrix = np.c_[gene_names, gene_counts_matrix]
    # Convert to data frame
    gene_counts_df = pd.DataFrame(
        gene_counts_matrix, 
        columns=["gene_symbol"] + ["sample%d" % (sn + 1) for sn in range(num_samples)]
    )

    # Generate random gene annotation. Gene annotations are usually long-format, with
    # one column for gene identifier and then one column per parameter, e.g.:
    # gene_symbol   gene_size       gene_type           ...
    #   gene1       10001           protein_coding      ...
    #   gene2       20002           lncRNA              ...
    #   ...         ...             ...                 ...
    #   geneN       50005           protein_coding      ...
    logging.info("\tPreparing gene annotation data..")
    gene_annotation = pd.DataFrame({
        "gene_symbol": ["gene_%d" % g for g in range(num_genes)],
        "gene_size": np.random.randint(low=100, high=100000, size=num_genes),  # Genes vary greatly in size
        "gene_type": generate_list_of_strings(["protein_coding", "lncRNA", "pseudogene"], [50.0, 20.0, 30.0], num_genes)
    })

    # Generate random sample annotation. Long-format with one row per sample and one column
    # for each parameter, e.g.:
    # sample_name   sample_type
    #   sample1         normal
    #   sample2         cancer
    #   ...             ...
    #   sampleN         cancer
    logging.info("\tPreparing sample information data..")
    sample_annotation = pd.DataFrame({
        "sample_name": ["sample%d" % (sn + 1) for sn in range(num_samples)],
        "sample_type": generate_list_of_strings(strings=["normal", "cancer"], percentage_of_each_string=[30.0, 70.0], length=num_samples)
    })

    return gene_counts_df, gene_annotation, sample_annotation


def merge_dgea_results_with_gene_annotation(dgea_results, gene_annotation):
    """
    Combines differential gene expression analysis (DGEA) results with gene annotation data, resulting
    in a single dataset with fold changes, p-values, gene type and gene name for each gene. Data sets
    will be merged on a common column named gene_symbol.

    Parameters
    ----------
    dgea_results : pd.DataFrame
        Dataframe formatted as DESeq2 output: Long-format, with one row per gene, and columns for
        gene_symbol, baseMean, log2FoldChange, and adjusted p-value (padj).
    gene_annotation : pd.DataFrame
        Dataframe with gene annotation data in long-format, containing one row per gene, with columns
        gene_type and gene_symbol.

    Returns
    -------
    pd.dataFrame
        Dataframe in long-format, with one row per gene, and columns for gene_symbol, baseMean, log2FoldChange,
        padj, and gene_type.
    """
    logging.info("Merging DGEA results and gene annotations..")

    try:
        merged = dgea_results.merge(gene_annotation, on="gene_symbol", how="inner")
    except KeyError:
        logging.error("Unable to merge DGEA results with gene annotations. One of the datasets appear to missing the 'gene_symbol' column.")
        sys.exit(-1)

    # Merging can change the size of the original data set. E.g. annotations can include multiple entries
    # per gene, or gene symbols in the DGEA results can be missing from the annotation.
    num_genes_difference = len(dgea_results) - len(merged)
    if num_genes_difference > 0:
        logging.warning("%d genes lost during gene annotation merge." % abs(num_genes_difference))
    elif num_genes_difference < 0:
        logging.warnning("%d entries possibly duplicated during merge (%d deduplicated)" % (abs(num_genes_difference), merged.gene_symbol.nunique()))

    return merged


def create_directory(directory_path):
    """
    Helper function that creates missing directory or directories in the specified path.

    Parameters
    ----------
    directory_path : Str
        Path to the directory or directories to be created.
    """
    if not os.path.exists(directory_path) and not directory_path == "":
        logging.info("Creating output directory %s" % directory_path)
        os.makedirs(directory_path)


def setup_logging(logfile_path):
    """
    Helper function to setup basic logging functionality. Writes logs to stdout and to a specified
    log file. Missing directories in the provided path will be created.

    Parameters
    ----------
    logfile_path : Str
        Path to the log file.
    """
    # Create directory if it doesn't already exist
    create_directory(os.path.dirname(logfile_path))

    # Setup logging to file and to stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-5.10s [%(asctime)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(logfile_path)),
            logging.StreamHandler()
        ]
    )


def draw_volcano_plot(data, axes, outpath=None, xlim=None, ylim=None):
    """
    Draws volcano plots (https://en.wikipedia.org/wiki/Volcano_plot_(statistics)) highlighting
    significantly differentially expressed genes with fold changes on the X-axis and p-values
    on the Y-axis.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format dataframe with one row for each gene and columns for log2FoldChange and padj,
        representing the log2-transformed fold change and the adjusted p-value for each gene.
    axes : matplotlib.axes._subplots.AxesSubplot
        An Axes object to which to add the generated plot.
    outpath : Str
        An optional path describing where to save the plot.
    xlim : tuple(float, float)
        Minimum and maximum value defining the range of the X-axis.
    ylim : tuple(float, float)
        Minimum and maximum value defining the range of the Y-axis.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Axes object containing the volcano plot
    """
    logging.info("Drawing volcano plot")

    # Plot configuration
    significance_palette = {"Not significant": "#bdc3c7", "Up-regulated": "#e74c3c", "Down-regulated": "#3498db"}  # Color dots by significance level
    dot_size = 30.0  # The size of each dot drawn on the plot
    missing_indicators_size = dot_size/1.5  # Arrows are a bit smaller than the dots
    missing_indicators_offset = 0.01  # Arrows can be cut off a bit if they're drawn right at the axis boundaries, so we offset them just a tiny bit
    sig_threshold_pvalue = 0.05
    sig_threshold_downregulated = -1.0
    sig_threshold_upregulated = 1.0

    # We transform p-values to -log10, so smaller p-values are higher up on the plot. This can
    # crash if there are NaNs in the data, so we'll replace any NaNs by p-values equal to 1.0.
    number_of_nans = sum(data.padj.isna())
    if number_of_nans > 0:
        data.padj = data.padj.fillna(1.0)
        logging.warning("%d p-values were changed to 1.0 due to missing values." % number_of_nans)
    data["p_neglog10"] = -np.log10(data.padj)

    # Mark genes according to our standard significance criteria:
    # - P-values less than 0.05
    # - Fold change greater than 2.0 or less than -2.0
    data["significant"] = "Not significant"
    data.loc[(data.padj < sig_threshold_pvalue) & (data.log2FoldChange > sig_threshold_upregulated), "significant"] = "Up-regulated"
    data.loc[(data.padj < sig_threshold_pvalue) & (data.log2FoldChange < sig_threshold_downregulated), "significant"] = "Down-regulated"

    # Draw volcano plot
    volcano_plot = sns.scatterplot(
        data=data,
        x="log2FoldChange",
        y="p_neglog10",
        hue="significant",
        palette=significance_palette,
        s=dot_size,
        ax=axes
    )
    volcano_plot.set_xlim(xlim)
    volcano_plot.set_ylim(ylim)
    volcano_plot.set_ylabel("Adjusted p-value (-log10)")
    volcano_plot.set_xlabel("Fold change (log2)")
    volcano_plot.spines["top"].set_visible(False)
    volcano_plot.spines["right"].set_visible(False)
    volcano_plot.axhline(-np.log10(sig_threshold_pvalue), linestyle="--", linewidth=1, color="black")
    volcano_plot.axvline(sig_threshold_downregulated, linestyle="--", linewidth=1, color="black")
    volcano_plot.axvline(sig_threshold_upregulated, linestyle="--", linewidth=1, color="black")

    # Handle axis limitations are provided, some genes may be drawn outside of the defined range.
    # We should indicate that there are genes present beyond the boundaries, and we'll do that by
    # drawing arrows pointing towards the out-of-bounds data points
    if xlim:
        min_x = xlim[0]
        max_x = xlim[1]
        # If limitations are set on the X-axis, find the data points that are currently excluded
        # from the plot.
        too_large_x = data.loc[data.log2FoldChange > max_x]
        too_small_x = data.loc[data.log2FoldChange < min_x]
        # Add indicators to the plot for these missing data points
        sns.scatterplot(x=max_x-missing_indicators_offset, y="p_neglog10", marker=">", data=too_large_x, hue="significant", palette=significance_palette, s=missing_indicators_size, legend=False, ax=axes)
        sns.scatterplot(x=min_x+missing_indicators_offset, y="p_neglog10", marker="<", data=too_small_x, hue="significant", palette=significance_palette, s=missing_indicators_size, legend=False, ax=axes)
    # Same for limits on the Y-axis
    if ylim:
        min_y = ylim[0]
        max_y = ylim[1]
        too_large_y = data.loc[data.p_neglog10 > max_y]
        too_small_y = data.loc[data.p_neglog10 > min_y]
        # Add indicators to the plot
        sns.scatterplot(x="log2FoldChange", y=max_y-missing_indicators_offset, marker="^", data=too_large_y, hue="significant", palette=significance_palette, s=missing_indicators_size, legend=False, ax=axes)
        sns.scatterplot(x="log2FoldChange", y=min_y+missing_indicators_offset, marker="v", data=too_small_y, hue="significant", palette=significance_palette, s=missing_indicators_size, legend=False, ax=axes)

    # Legend is self-explanatory without the title, so we remove it
    volcano_plot.legend().set_title(None)

    # Save or show
    if outpath:
        logging.info("Saving volcano plot to %s" % outpath)
        create_directory(os.path.dirname(outpath))
        fig = volcano_plot.get_figure()
        # This is a bit hacky and depends on the fig size used, but it limits
        # the scope to only the volcano plot.
        extent = axes.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(outpath, bbox_inches=extent.expanded(1.1, 1.3))

    return volcano_plot


def draw_boxplot_of_gene_expression(df, axes, outpath=None):
    """
    Draws basic box plot of gene expression between conditions for a selection of genes.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format dataframe with one row per gene-sample combination. Must include columns
        "gene_symbol", "sample_type", and "transcripts_per_million".
    axes : matplotlib.axes._subplots.AxesSubplot
        An Axes object to which to add the generated plot.
    outpath : Str
        An optional path describing where to save the plot.
    """
    logging.info("Drawing boxplot")

    boxplot = sns.boxplot(
        x="gene_symbol",
        y="transcripts_per_million",
        hue="sample_type",
        palette={"normal": "#95a5a6", "cancer": "#9b59b6"},
        data=df,
        ax=axes
    )
    plt.yscale("log")
    plt.ylabel("Transcripts per million (log2)")
    plt.xlabel("Gene symbol")
    boxplot.spines["top"].set_visible(False)
    boxplot.spines["right"].set_visible(False)
    boxplot.legend().set_title(None)

    if outpath:
        logging.info("Saving boxplot to %s" % outpath)
        create_directory(os.path.dirname(outpath))
        fig = boxplot.get_figure()
        # These are a bit awkward, and will depend on the fig size used. But we manually limit
        # the boundaries to only include the boxplot
        extent = axes.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        extent.y0 = extent.y0 / 2
        extent.y1 = extent.y1 * 1.01
        extent.x0 = extent.x0 / 2
        extent.x1 = extent.x1 * 1.01
        fig.savefig(outpath, bbox_inches=extent)
    
    return boxplot


def tpm_from_counts_matrix(gene_counts_matrix, gene_annotation):
    """
    Computes transcripts per million (TPM) for each gene in each sample. TPM normalization
    ensures each sample's total expression adds up to exactly one million.

    Parameters
    ----------
    gene_counts_matrix : pd.DataFrame
        Data frame with one column for gene IDs and the remaining columns representing
        samples, one column per sample.
    gene_annotation: pd.DataFrame
        Data frame with one column for gene ID and one column for each gene parameter, e.g.
        gene type, gene size, gene position.
    
    Returns
    -------
    pd.DataFrame
        Long-format data frame with one column for gene ID, one column for sample name, and
        one column for normalized gene expression (TPM).
    """
    logging.info("Calculating transcripts per million (TPM)")

    # In order to merge our counts matrix with the gene annotation, the counts matrix
    # must be trasnformed to long-format
    df = gene_counts_matrix.melt(
        id_vars = "gene_symbol",
        value_vars = [c for c in gene_counts_matrix.columns if "sample" in c],
        var_name="sample_name",
        value_name="gene_counts"
    )
    # Ensure gene counts are integers
    df.gene_counts = df.gene_counts.astype(int)

    # Merge gene sizes from the annotation into the expression data frame
    df = df.merge(gene_annotation, on="gene_symbol", how="left")

    # Calculate transcripts per million (TPM) for each gene in each sample
    df["reads_per_kilobase"] = df.gene_counts / (df.gene_size / 1000.0)
    df["per_million_scaling_factor"] = df.groupby("sample_name").reads_per_kilobase.transform(sum) / 1000000.0
    df["transcripts_per_million"] = df.reads_per_kilobase / df.per_million_scaling_factor

    # Make sure all samples now add up to 1M
    df["sample_sum_tpm"] = df.groupby("sample_name").transcripts_per_million.transform("sum")
    #print(df.sample_sum_tpm.describe())
    if not int(df.drop_duplicates("sample_name").sample_sum_tpm.sum()) == int((df.sample_name.nunique() * 10**6)):
        logging.warning("All samples' total TPM does not add up to one million per sample")
    
    return df[["sample_name", "gene_symbol", "transcripts_per_million"]]


def demo():
    """
    Runs this script in demo mode, using dummy data as input so one can get the impression
    of what it does without having to provide real input data.
    """
    
    ###################################################
    ### Setup data and visualization configurations ###
    ###################################################
    output_directory = os.getcwd()
    plot_outpath = os.path.join(output_directory, "example_plot.pdf")

    logfile_path = os.path.join(output_directory, "log.txt")
    setup_logging(logfile_path)
    logging.info("Logs will be saved to %s" % logfile_path)

    # Read results from differential gene expression analysis (DGEA). These results contain
    # fold changes and p-values. The fold changes represent the difference in gene expression,
    # or gene activity, between two conditions. Here, we've compared genes from normal samples
    # with genes from cancer samples. Normal samples serve as the reference, which means that
    # fold changes represent the difference from normal samples as observed in cancer samples,
    # e.g. a fold change of 2 means the gene is twice as highly expressed in cancer samples.
    dgea_results_path = "../input_data/dgea_example_data.csv"
    logging.info("Reading differetial gene expression analysis (DGEA) results from %s" % dgea_results_path)
    dgea_results = pd.read_csv(dgea_results_path)

    # Generate read counts matrix, gene annotation and sample annotation. Read counts matrices
    # represent measured gene activity, typically represented as one integer per gene per sample.
    # Specifically, the integer represents the number of sequence reads that align to the gene in 
    # the respective sample. 
    # Gene annotations provide information about each gene, e.g. its canonical symbol, its size,
    # position in the genome, etc.
    # Sample annotations provide information about each sample, e.g. its identifier and the type
    # of tissue the sample is derived from.
    gene_counts, gene_annotation, sample_annotation = generate_dummy_data()

    # Calculate normalized expression estimates for each gene in each sample. Raw read counts do
    # not accurately represent the gene activity in a sample, due to biases inherent to the sequencing
    # procedure. For one, different samples have different total number of reads. If one sample has
    # twice the total number of reads as another sample, any gene in the sample will have twice the
    # measured expression. Further, a gene that is twice the size of another gene will need to have
    # twice as many reads aligned to it to represent the same level of expression. To account for
    # these differences, we need to normalize the number of reads for the total number of reads in
    # a sample (also known as the sample's sequencing depth), and for the size of each gene. 
    # There are many ways to do this, here we use a metric called Transcripts Per Million (TPM).
    # TPM adjusts each gene's read count for the length of the gene, and also adjusts gene expression
    # values in every sample to sum to exactly 1 million. This makes it easier to compare
    # numbers between samples, since every gene expression value is a fraction of 1 million, instead
    # of a fraction of a different total in every sample.
    tpm = tpm_from_counts_matrix(gene_counts_matrix=gene_counts, gene_annotation=gene_annotation)
    # Merge in sample annotation with experimental condition (cancer vs normal)
    tpm = tpm.merge(sample_annotation, on="sample_name")

    # Prepare a figure. We'll have two plots eventually, one volcano plot of the DGEA results
    # and one boxplot showing the difference in gene expression between conditions for the
    # the most varying genes.
    fig, axs = plt.subplots(2, figsize=(10, 8))

    # Finally, config plot sizes and font sizes
    fontsize_big = 12
    fontsize_medium = 10
    fontsize_small = 8
    plt.rcParams.update({
        "font.family": "Helvetica",
        "xtick.labelsize": fontsize_small,
        "ytick.labelsize": fontsize_small,
        "axes.labelsize": fontsize_medium,
        "axes.titlesize": fontsize_big,
        "figure.figsize": [6.27, 4.45]
    })

    ###########################################################
    ### Draw volcano plot of differentially expressed genes ###
    ###########################################################

    # We're only interested in DGEA results from genes of a certain type. In order
    # to know which type a gene is, we need to merge the gene annotation data and
    # the DGEA results data
    merged = merge_dgea_results_with_gene_annotation(dgea_results, gene_annotation)
    original_size = len(merged)
    merged = merged.loc[merged.gene_type.isin(["protein_coding", "lincRNA"])]
    logging.info("%d -> %d genes after filtering on gene type" % (original_size, len(merged)))

    # Draw volcano plot. Volcano plots show differences in expression between conditions on
    # the X-axis (represented as fold-changes) and statistical significance of the observed 
    # difference on the Y-axis. Each dot represents a gene.
    volcano_plot = draw_volcano_plot(merged, axes=axs[0], xlim=(-3, 3))

    #################################################################
    ### Plot the expression of the differentially expressed genes ###
    #################################################################

    # Plot a few of the most differentially expressed genes. We'll draw simple boxplots of
    # the normalized expression values for each gene in each condition. We can't plot all 15,000
    # genes, so let's do the five genes that show the highest up-regulation in cancer.
    significance_threshold = 0.05
    num_genes = 5
    dgea_results_sorted = dgea_results.loc[dgea_results.padj < significance_threshold].sort_values(by="log2FoldChange", ascending=False)
    top_up_regulated_genes = dgea_results_sorted.head(num_genes).gene_symbol.tolist()
    boxplot_data = tpm.loc[tpm.gene_symbol.isin(top_up_regulated_genes)]
    # We're working on randomly generated data, which means there's not necessarily any noticable
    # differences between conditions in our fabricated expression data. For the boxplots to make
    # sense, we cheat a bit and make the values in cancer samples higher than those in normal samples.
    boxplot_data.loc[boxplot_data.sample_type == "normal", "transcripts_per_million"] = boxplot_data.loc[boxplot_data.sample_type == "normal"].transcripts_per_million / 5
    boxplot = draw_boxplot_of_gene_expression(df=boxplot_data, axes=axs[1])
    
    # Finish up figure
    fig.tight_layout()

    # Save or show
    if plot_outpath:
        logging.info("Saving plot to %s" % plot_outpath)
        plt.savefig(plot_outpath)
    else:
        logging.info("No outpath provded for plot.")
        plt.show()
    
    # Done!
    logging.info("Bleep boop. Done!")


if __name__ == "__main__":
    import random
    random.seed(1234)
    demo()
