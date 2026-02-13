import matplotlib.pyplot as plt
import json
from typing import Dict, Any, Optional

class ChartStyler:
    """
    Utility class for applying dynamic styling to matplotlib charts.
    Supports themes, colors, fonts, and layout customization via JSON config.
    """
    
    # Pre-defined color palettes
    PALETTES = {
        'medical': {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'background': '#FFFFFF',
            'text': '#333333'
        },
        'dark': {
            'primary': '#00D9FF',
            'secondary': '#FF6B9D',
            'accent': '#FFE66D',
            'background': '#1a1a1a',
            'text': '#FFFFFF'
        },
        'colorblind': {
            'primary': '#0173B2',
            'secondary': '#DE8F05',
            'accent': '#029E73',
            'background': '#FFFFFF',
            'text': '#000000'
        },
        'vibrant': {
            'primary': '#EE7733',
            'secondary': '#0077BB',
            'accent': '#33BBEE',
            'background': '#FFFFFF',
            'text': '#000000'
        }
    }
    
    def __init__(self, styling_json: Optional[Any] = None):
        """
        Initialize styler with optional JSON configuration.
        
        Args:
            styling_json: JSON string OR Dictionary with styling configuration
        """
        if styling_json:
            raw_config = {}
            if isinstance(styling_json, dict):
                raw_config = styling_json
            elif isinstance(styling_json, str):
                try:
                    raw_config = json.loads(styling_json)
                except json.JSONDecodeError:
                    print(f"⚠️ Invalid styling JSON, using defaults")
                    raw_config = {}
            
            # Normalization Logic (Flat -> Nested)
            self.config = {}
            
            # 1. Preserve existing structure if valid
            if 'style' in raw_config: self.config['style'] = raw_config['style']
            if 'colors' in raw_config: self.config['colors'] = raw_config['colors']
            if 'legend' in raw_config: self.config['legend'] = raw_config['legend']
            if 'figure' in raw_config: self.config['figure'] = raw_config['figure']
            if 'plot' in raw_config: self.config['plot'] = raw_config['plot']

            # 2. Map flat keys if not already present
            self.config.setdefault('style', {})
            self.config.setdefault('colors', {})
            
            # Theme
            if 'theme' in raw_config: 
                self.config['style']['theme'] = raw_config['theme']
            
            # Fonts/Sizes
            for key in ['title_size', 'label_size', 'tick_size', 'font_family', 'font_weight', 'grid', 'xlim', 'ylim', 'spines']:
                if key in raw_config:
                     self.config['style'][key] = raw_config[key]
            
            # Labels / Text
            if 'title' in raw_config: self.config['text'] = {'title': raw_config['title']}
            if 'xlabel' in raw_config: self.config.setdefault('text', {})['xlabel'] = raw_config['xlabel']
            if 'ylabel' in raw_config: self.config.setdefault('text', {})['ylabel'] = raw_config['ylabel']
            
            # Ticks (Renaming)
            if 'xtick_labels' in raw_config: self.config.setdefault('ticks', {})['x_labels'] = raw_config['xtick_labels']
            if 'ytick_labels' in raw_config: self.config.setdefault('ticks', {})['y_labels'] = raw_config['ytick_labels']
            
            # Colors
            if 'primary_color' in raw_config: self.config['colors']['primary'] = raw_config['primary_color']
            if 'color' in raw_config: self.config['colors']['primary'] = raw_config['color']
            if 'bar_color' in raw_config: self.config['colors']['primary'] = raw_config['bar_color']
            if 'line_color' in raw_config: self.config['colors']['primary'] = raw_config['line_color']
            if 'background_color' in raw_config: self.config['colors']['background'] = raw_config['background_color']
            
        else:
            self.config = {}
    
    def apply(self, fig, ax):
        """
        Apply styling configuration to matplotlib figure and axis.
        
        Args:
            fig: Matplotlib figure object
            ax: Matplotlib axis object
        """
        # Apply theme first (sets base colors)
        if 'style' in self.config and 'theme' in self.config['style']:
            self._apply_theme(fig, ax, self.config['style']['theme'])
        
        # Apply custom colors (overrides theme)
        if 'colors' in self.config:
            self._apply_colors(fig, ax, self.config['colors'])
        
        # Apply style settings
        if 'style' in self.config:
            self._apply_style_settings(ax, self.config['style'])
            
            # Apply font settings if present
            self._apply_fonts(ax, self.config['style'])
            
            # Apply axis limits
            self._apply_limits(ax, self.config['style'])

        # Apply specific plot settings (markers, lines)
        self._apply_plot_elements(ax)
        
        # Apply text content (Title, Labels)
        if 'text' in self.config:
            self._apply_text_content(ax, self.config['text'])
            
        # Apply tick updates (renaming categories)
        if 'ticks' in self.config:
            self._apply_tick_labels(ax, self.config['ticks'])
        
        # Apply legend settings
        if 'legend' in self.config:
            self._apply_legend(ax, self.config['legend'])
            
        # Apply figure settings
        if 'figure' in self.config:
            self._apply_figure_settings(fig, self.config['figure'])

    def _apply_text_content(self, ax, text_config: Dict[str, str]):
        """Apply custom text content (title, labels)."""
        if 'title' in text_config:
            ax.set_title(text_config['title'])
        if 'xlabel' in text_config:
            ax.set_xlabel(text_config['xlabel'])
        if 'ylabel' in text_config:
            ax.set_ylabel(text_config['ylabel'])
            
    def _apply_tick_labels(self, ax, tick_config: Dict[str, Any]):
        """Apply custom tick labels (mapping or list)."""
        # X-Axis Ticks
        if 'x_labels' in tick_config:
            labels = tick_config['x_labels']
            if isinstance(labels, dict):
                # Mapping: Get current labels, replace matches
                current_ticks = ax.get_xticklabels()
                new_labels = [labels.get(t.get_text(), t.get_text()) for t in current_ticks]
                ax.set_xticklabels(new_labels)
            elif isinstance(labels, list):
                # Direct replacement (careful with length)
                if len(labels) == len(ax.get_xticks()):
                    ax.set_xticklabels(labels)
                    
        # Y-Axis Ticks
        if 'y_labels' in tick_config:
            labels = tick_config['y_labels']
            if isinstance(labels, dict):
                current_ticks = ax.get_yticklabels()
                new_labels = [labels.get(t.get_text(), t.get_text()) for t in current_ticks]
                ax.set_yticklabels(new_labels)
            elif isinstance(labels, list):
                if len(labels) == len(ax.get_yticks()):
                    ax.set_yticklabels(labels)

    def _apply_theme(self, fig, ax, theme: str):
        """Apply a pre-defined theme."""
        if theme in self.PALETTES:
            palette = self.PALETTES[theme]
            
            # Set background colors
            fig.patch.set_facecolor(palette['background'])
            ax.set_facecolor(palette['background'])
            
            # Set text colors
            ax.tick_params(colors=palette['text'], which='both')
            ax.xaxis.label.set_color(palette['text'])
            ax.yaxis.label.set_color(palette['text'])
            ax.title.set_color(palette['text'])
            
            # Set spine colors
            for spine in ax.spines.values():
                spine.set_edgecolor(palette['text'])
                spine.set_alpha(0.3)
        else:
            # Try to apply as standard matplotlib/seaborn style
            try:
                import seaborn as sns
                # Check directly if it's a known style or available in plt.style
                if theme in plt.style.available or theme in ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']:
                    try:
                        # Try seaborn first for grid styles
                        if theme in ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']:
                            sns.set_style(theme)
                        else:
                            plt.style.use(theme)
                        # pine_logger(f"✅ Applied standard theme '{theme}'")
                    except:
                        plt.style.use(theme)
                else:
                    print(f"⚠️ Unknown theme '{theme}', available: {list(self.PALETTES.keys())} + standard matplotlib styles")
            except Exception as e:
                print(f"⚠️ Failed to apply theme '{theme}': {e}")
    
    def _apply_colors(self, fig, ax, color_config: Dict[str, Any]):
        """Apply custom color configuration."""
        # Apply palette if specified
        if 'palette' in color_config:
            palette_name = color_config['palette']
            if palette_name in self.PALETTES:
                palette = self.PALETTES[palette_name]
                fig.patch.set_facecolor(palette['background'])
                ax.set_facecolor(palette['background'])
        
        # Apply primary color to lines/bars
        if 'primary' in color_config:
            # This will be used by plot functions if set before plotting
            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[color_config['primary']])
            # Also try to update existing lines
            for line in ax.get_lines():
                line.set_color(color_config['primary'])
            
            # Update patches (bars, histograms)
            for patch in ax.patches:
                patch.set_facecolor(color_config['primary'])

        
        # Apply background color
        if 'background' in color_config:
            fig.patch.set_facecolor(color_config['background'])
            ax.set_facecolor(color_config['background'])
            
    def _apply_fonts(self, ax, style_config: Dict[str, Any]):
        """Apply font settings."""
        if 'font_family' in style_config:
            plt.rcParams['font.family'] = style_config['font_family']
            ax.title.set_fontfamily(style_config['font_family'])
            ax.xaxis.label.set_fontfamily(style_config['font_family'])
            ax.yaxis.label.set_fontfamily(style_config['font_family'])
            for item in ax.get_xticklabels() + ax.get_yticklabels():
                item.set_fontfamily(style_config['font_family'])

        if 'font_weight' in style_config:
            ax.title.set_fontweight(style_config['font_weight'])
            ax.xaxis.label.set_fontweight(style_config['font_weight'])
            ax.yaxis.label.set_fontweight(style_config['font_weight'])

    def _apply_limits(self, ax, style_config: Dict[str, Any]):
        """Apply axis limits."""
        if 'xlim' in style_config:
            ax.set_xlim(style_config['xlim'])
        if 'ylim' in style_config:
            ax.set_ylim(style_config['ylim'])

    def _apply_style_settings(self, ax, style_config: Dict[str, Any]):
        """Apply style settings like font sizes, grid, etc."""
        # Title size
        if 'title_size' in style_config:
            ax.title.set_fontsize(style_config['title_size'])
        
        # Label size
        if 'label_size' in style_config:
            ax.xaxis.label.set_fontsize(style_config['label_size'])
            ax.yaxis.label.set_fontsize(style_config['label_size'])
            ax.tick_params(labelsize=style_config.get('tick_size', style_config['label_size'] * 0.8))
        
        # Grid
        if 'grid' in style_config:
            grid_conf = style_config['grid']
            if isinstance(grid_conf, bool):
                if grid_conf:
                    ax.grid(True, alpha=0.3)
                else:
                    ax.grid(False)
            elif isinstance(grid_conf, dict):
                ax.grid(True, **grid_conf)
        
        # Spine visibility
        if 'spines' in style_config:
            for spine_name, visible in style_config['spines'].items():
                if spine_name in ax.spines:
                    ax.spines[spine_name].set_visible(visible)
                    
    def _apply_plot_elements(self, ax):
        """Apply styling to plot elements like lines and markers."""
        if 'plot' not in self.config:
            return
            
        plot_config = self.config['plot']
        
        # Update existing lines
        for line in ax.get_lines():
            if 'linewidth' in plot_config:
                line.set_linewidth(plot_config['linewidth'])
            if 'linestyle' in plot_config:
                line.set_linestyle(plot_config['linestyle'])
            if 'marker' in plot_config:
                line.set_marker(plot_config['marker'])
            if 'markersize' in plot_config:
                line.set_markersize(plot_config['markersize'])
                
    def _apply_figure_settings(self, fig, figure_config: Dict[str, Any]):
        if 'figsize' in figure_config:
            fig.set_size_inches(figure_config['figsize'])

    def _apply_legend(self, ax, legend_config: Dict[str, Any]):
        """Apply legend configuration."""
        legend = ax.get_legend()
        
        if legend:
            # Show/hide legend
            if 'show' in legend_config:
                legend.set_visible(legend_config['show'])
            
            # Position (Note: creating a new legend might be needed if strictly changing loc)
            if 'position' in legend_config and legend_config.get('show', True):
                # We can't easily move an existing legend's loc attribute effectively without recreating it
                # but we can try setting bbox_to_anchor if needed, or re-calling ax.legend
                # simpler approach: Pass standard loc strings
                pass 
                
    @staticmethod
    def get_theme_list():
        """Return list of available themes."""
        return list(ChartStyler.PALETTES.keys())
    
    @staticmethod
    def create_styling_json(
        theme: Optional[str] = None,
        primary_color: Optional[str] = None,
        title_size: int = 14,
        grid: bool = True,
        font_family: Optional[str] = None,
        linewidth: Optional[float] = None
    ) -> str:
        """
        Helper to create styling JSON with common parameters.
        """
        config = {
            "style": {
                "title_size": title_size,
                "grid": grid
            }
        }
        
        if theme:
            config["style"]["theme"] = theme
        
        if primary_color:
            config["colors"] = {"primary": primary_color}
            
        if font_family:
            config["style"]["font_family"] = font_family
            
        if linewidth:
             config.setdefault("plot", {})["linewidth"] = linewidth
        
        return json.dumps(config)

    @staticmethod
    def create_publication_style(
        theme: str = 'white',
        font_family: str = 'Arial',
        font_size: int = 12,
        dpi: int = 300
    ) -> str:
        """
        Create a publication-ready styling configuration (Nature/Science style).
        Features: High DPI, Arial font, minimal grid, no top/right spines.
        """
        config = {
            "style": {
                "theme": theme,
                "font_family": font_family,
                "title_size": font_size,
                "label_size": font_size,
                "tick_size": int(font_size * 0.8),
                "grid": False,
                "spines": {"top": False, "right": False},
                "dpi": dpi
            },
            "figure": {
                "figsize": [4, 4] # Standard single column width
            },
            "legend": {
                "show": True
            }
        }
        return json.dumps(config)
