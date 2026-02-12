import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from . import SelectionPipeline


class Volcano_selection(SelectionPipeline):
    """
    volcano plot.

    """

    def __init__(self,
                 k,
                 z_importance_threshold=1.,
                 strategy="fold",
                 p_threshold=0.05,
                 fc_threshold=2,
                 log_domain=False,
                 absolute=True,
                 target_label=1):
        """

        Args:
            strategy (str, optional): Choosing strategy. One of {"p" or "fold"} Defaults to "fold".
            p_threshold (float, optional): p-value threshold. Only feature has p-value higher than threshold will be considered. Defaults to 0.05.
            fc_threshold (int, optional): fold change threshold. Only feature has fold change higher than threshold will be considered. Defaults to 2.
            log_domain (bool, optional): Whether input data is in log_domain. Defaults to False.
            absolute (bool, optional): If true, then take absolute value on score while strategy == "p". Defaults to True.
            target_label : the target label.
        """
        super().__init__(k=k, z_importance_threshold=z_importance_threshold)
        self.strategy = strategy
        self.fc_threshold = fc_threshold
        self.p_threshold = p_threshold
        self.log_domain = log_domain
        self.absolute = absolute
        self.name = "Volcano Plot_" + self.strategy
        self.missing_value = 0
        self.target_label = target_label

    def Scoring(self, x, y):
        """
        Compute the fold change and p-value on each feature.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
           pandas.DataFrame: A dataframe records p-value and fold change.
        """
        positive = y == self.target_label
        negative = np.logical_not(positive)

        x = x.replace(0, np.nan)

        # fold change
        if not self.log_domain:
            log_fold = np.log2(x[positive].mean(axis=0) /
                               x[negative].mean(axis=0))
        else:
            log_fold = x[positive].mean(axis=0) - x[negative].mean(axis=0)

        # Welch t test:
        #     normal assumption
        #     diffirent sample size
        #     diffirent varience
        #     unpaired
        n_positive = x[positive].shape[0]
        n_negative = x[negative].shape[0]
        diff = x[positive].mean(axis=0) - x[negative].mean(axis=0)

        s_positive = ((x[positive] - x[positive].mean(axis=0))**
                      2).sum(axis=0) / (n_positive - 1)
        s_negative = ((x[negative] - x[negative].mean(axis=0))**
                      2).sum(axis=0) / (n_negative - 1)
        st = np.sqrt(s_positive / n_positive + s_negative / n_negative)

        t_statistic = np.abs(diff / st)
        df = (s_positive / n_positive + s_negative / n_negative)**2 / (
            (s_positive / n_positive)**2 / (n_positive - 1) +
            (s_negative / n_negative)**2 / (n_negative - 1))

        # 2 side testing
        #print("t statistic: ", t_statistic.min(), t_statistic.mean(), t_statistic.max())
        #print("degree of freedom: ", df.min(), df.mean(), df.max())

        p_value = t.cdf(x=-t_statistic, df=df) * 2
        log_p = -np.log10(p_value)

        scores = pd.DataFrame(
            {
                "log_p_value": log_p,
                "log_fold_change": log_fold
            },
            index=log_fold.index)
        return scores

    def Select(self, scores):
        """
        Choosing the features which has score higher than threshold in assigned strategy.

        If strategy == "fold": sort in fold change and return p-value

        If strategy == "p": sort in p-value and return fold change

        Args:
            scores (pandas.DataFrame): A dataframe records p-value and fold change.
            k (int): Number of features to select.

        Returns:
            pandas.Series: The score for k selected features in assigned strategy.
        """
        log_fold = scores["log_fold_change"]
        log_p = scores["log_p_value"]
        # choose fold change > 2 and p value < 0.05 in log scale
        significant = np.logical_and(
            np.abs(log_fold) >= np.log2(self.fc_threshold), log_p
            > -np.log10(self.p_threshold))
        self.significant = significant

        # choose top k logged p-value
        if self.strategy == "fold":
            selected = np.abs(log_fold).loc[significant].sort_values().tail(
                self.k)
            selected_score = pd.Series(log_p.loc[selected.index],
                                       index=selected.index,
                                       name=self.name)
        elif self.strategy == "p":
            selected = log_p.loc[significant].sort_values().tail(self.k)
            if self.absolute:
                selected_score = pd.Series(
                    np.abs(log_fold.loc[selected.index]),
                    index=selected.index,
                    name=self.name).sort_values(ascending=False)
            else:
                selected_score = pd.Series(
                    log_fold.loc[selected.index],
                    index=selected.index,
                    name=self.name).sort_values(ascending=False)
        else:
            raise "select_by must be one of {fold} or {p}"

        return selected_score

    def plotting(self,
                 external=False,
                 external_score=None,
                 title="Volcano Plot",
                 show=True,
                 saving=False,
                 save_path="./output/",
                 styling=None):
        """
        Enhanced Plotting with custom styling and labeling.

        Args:
            styling (dict/str, optional): Styling configuration containing:
                - colors: {'up': 'red', 'down': 'blue', 'ns': 'gray'}
                - style: {'theme': 'whitegrid', 'dpi': 300}
                - labels: {'top_n': 5}
        """
        import json
        from PineBioML.visualization.style import ChartStyler
        
        # Parse styling
        config = {}
        if styling:
            if isinstance(styling, str):
                try:
                    config = json.loads(styling)
                except:
                    config = {}
            else:
                config = styling
        
        # Extract styling parameters
        colors = config.get('colors', {})
        up_color = colors.get('up', '#E64B35')  # Default Red-ish
        down_color = colors.get('down', '#3C5488') # Default Blue-ish
        ns_color = colors.get('ns', 'gray')
        
        style_opts = config.get('style', {})
        dpi = style_opts.get('dpi', 150)
        
        label_opts = config.get('labels', {})
        top_n_labels = label_opts.get('top_n', 0)

        log_fold = self.scores["log_fold_change"]
        log_p = self.scores["log_p_value"]
        
        # Define masks
        sig_threshold = -np.log10(self.p_threshold)
        fc_threshold_log = np.log2(self.fc_threshold)
        
        mask_sig = log_p > sig_threshold
        mask_up = (log_fold >= fc_threshold_log) & mask_sig
        mask_down = (log_fold <= -fc_threshold_log) & mask_sig
        mask_ns = ~(mask_up | mask_down)

        # Create Plot
        plt.figure(figsize=config.get('figure', {}).get('figsize', (10, 7)))
        
        # Plot Non-Significant
        plt.scatter(x=log_fold[mask_ns],
                    y=log_p[mask_ns],
                    s=10,
                    alpha=0.5,
                    color=ns_color,
                    label='Not Significant')
        
        # Plot Up-Regulated
        plt.scatter(x=log_fold[mask_up],
                    y=log_p[mask_up],
                    s=20,
                    alpha=0.8,
                    color=up_color,
                    label=f'Up-regulated (FC > {self.fc_threshold})')
        
        # Plot Down-Regulated
        plt.scatter(x=log_fold[mask_down],
                    y=log_p[mask_down],
                    s=20,
                    alpha=0.8,
                    color=down_color,
                    label=f'Down-regulated (FC < -{self.fc_threshold})')

        # Add Labels for Top N Significant Genes if requested
        if top_n_labels > 0:
            # Combine up and down, sort by p-value (descending log_p)
            # Create a dataframe for easy sorting
            sig_df = pd.DataFrame({'log_fold': log_fold, 'log_p': log_p})
            sig_df = sig_df[mask_up | mask_down]
            top_genes = sig_df.sort_values('log_p', ascending=False).head(top_n_labels)
            
            texts = []
            for gene, row in top_genes.iterrows():
                texts.append(plt.text(row['log_fold'], row['log_p'], str(gene), fontsize=9))
            
            # Simple adjustment to avoid overlap (basic implementation)
            # For production, adjustText library is better but adds dependency
            pass

        # Apply ChartStyler for theme/grid/fonts
        if styling:
            styler = ChartStyler(styling)
            styler.apply(plt.gcf(), plt.gca())
        
        # Customize Axes if not handled by Styler
        plt.title(title)
        plt.xlabel("Log2 Fold Change")
        plt.ylabel("-Log10 P-value")
        
        # Add Threshold Lines
        plt.axvline(fc_threshold_log, linestyle="--", color="black", alpha=0.3, linewidth=0.8)
        plt.axvline(-fc_threshold_log, linestyle="--", color="black", alpha=0.3, linewidth=0.8)
        plt.axhline(sig_threshold, linestyle="--", color="black", alpha=0.3, linewidth=0.8)
        
        plt.legend(loc='best')
        plt.tight_layout()

        if saving:
            # Append extension if missing, verify path
            full_path = save_path if save_path.endswith('.png') else f"{save_path}.png"
            plt.savefig(full_path, format="png", dpi=dpi, bbox_inches='tight')
            
        if show:
            plt.show() # Note: In Agg backend this does nothing, which is fine
        
        # Explicitly close to free memory
        plt.close()
