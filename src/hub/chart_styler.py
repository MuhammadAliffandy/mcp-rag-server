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
    
    def __init__(self, styling_json: Optional[str] = None):
        """
        Initialize styler with optional JSON configuration.
        
        Args:
            styling_json: JSON string with styling configuration
        """
        if styling_json:
            try:
                self.config = json.loads(styling_json)
            except json.JSONDecodeError:
                print(f"⚠️ Invalid styling JSON, using defaults")
                self.config = {}
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
        
        # Apply legend settings
        if 'legend' in self.config:
            self._apply_legend(ax, self.config['legend'])
    
    def _apply_theme(self, fig, ax, theme: str):
        """Apply a pre-defined theme."""
        if theme in self.PALETTES:
            palette = self.PALETTES[theme]
            
            # Set background colors
            fig.patch.set_facecolor(palette['background'])
            ax.set_facecolor(palette['background'])
            
            # Set text colors
            ax.tick_params(colors=palette['text'])
            ax.xaxis.label.set_color(palette['text'])
            ax.yaxis.label.set_color(palette['text'])
            ax.title.set_color(palette['text'])
            
            # Set spine colors
            for spine in ax.spines.values():
                spine.set_edgecolor(palette['text'])
                spine.set_alpha(0.3)
        else:
            print(f"⚠️ Unknown theme '{theme}', available: {list(self.PALETTES.keys())}")
    
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
            # This will be used by plot functions
            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[color_config['primary']])
        
        # Apply background color
        if 'background' in color_config:
            fig.patch.set_facecolor(color_config['background'])
            ax.set_facecolor(color_config['background'])
    
    def _apply_style_settings(self, ax, style_config: Dict[str, Any]):
        """Apply style settings like font sizes, grid, etc."""
        # Title size
        if 'title_size' in style_config:
            ax.title.set_fontsize(style_config['title_size'])
        
        # Label size
        if 'label_size' in style_config:
            ax.xaxis.label.set_fontsize(style_config['label_size'])
            ax.yaxis.label.set_fontsize(style_config['label_size'])
        
        # Grid
        if 'grid' in style_config:
            ax.grid(style_config['grid'], alpha=0.3)
        
        # Spine visibility
        if 'spines' in style_config:
            for spine_name, visible in style_config['spines'].items():
                if spine_name in ax.spines:
                    ax.spines[spine_name].set_visible(visible)
    
    def _apply_legend(self, ax, legend_config: Dict[str, Any]):
        """Apply legend configuration."""
        legend = ax.get_legend()
        
        if legend:
            # Show/hide legend
            if 'show' in legend_config:
                legend.set_visible(legend_config['show'])
            
            # Position
            if 'position' in legend_config and legend_config.get('show', True):
                ax.legend(loc=legend_config['position'])
    
    @staticmethod
    def get_theme_list():
        """Return list of available themes."""
        return list(ChartStyler.PALETTES.keys())
    
    @staticmethod
    def create_styling_json(
        theme: Optional[str] = None,
        primary_color: Optional[str] = None,
        title_size: int = 14,
        grid: bool = True
    ) -> str:
        """
        Helper to create styling JSON with common parameters.
        
        Args:
            theme: Theme name (medical, dark, colorblind, vibrant)
            primary_color: Primary color hex code
            title_size: Title font size
            grid: Show grid
        
        Returns:
            JSON string with styling configuration
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
        
        return json.dumps(config)
