import functools
import os.path
import time

from matplotlib import font_manager
import settings


class FontHelper:

    def __init__(self):
         fontfamily_text = 'data/ressources/fonts/OpenSans-Regular.ttf'
         opensans = font_manager.FontProperties(fname=fontfamily_text)
         opensans._size = 12
         # opensans._size = 12
         self.sub_text_font = opensans
    
         fontfamily_title = 'data/ressources/fonts/Montserrat-SemiBold.ttf'
         montserrat = font_manager.FontProperties(fname=fontfamily_title)
         montserrat._size = 24
         # montserrat._size = 26
         montserrat._weight = 'bold'
         self.title_font = montserrat

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def custom_font(font_size=12, font_weight='normal', font_name='Open Sans'):

        start_time = time.time()
        if font_name == 'Open Sans':
            fontfamily_text = "data/ressources/fonts/OpenSans-Regular.ttf"
        elif font_name == 'proxima-italic':
            fontfamily_text = 'data/ressources/fonts/proxima-italic.ttf'
        elif font_name == 'Open Sans Bold':
            fontfamily_text = 'data/ressources/fonts/OpenSans.ttf'
        elif font_name == 'Montserrat Medium':
            fontfamily_text = 'data/ressources/fonts/Montserrat-Medium.ttf'
        elif font_name == 'Montserrat Regular':
            fontfamily_text = 'data/ressources/fonts/Montserrat-Regular.ttf'

        elif os.path.exists('data/ressources/fonts/{font_name}.ttf'):
            fontfamily_text = 'data/ressources/fonts/{font_name}.ttf'

        elif os.path.exists('data/ressources/fonts/{font_name}.otf'):
            fontfamily_text = 'data/ressources/fonts/{font_name}.otf'

        else:
            fontfamily_text = 'data/ressources/fonts/Montserrat-SemiBold.ttf'

        opensans = font_manager.FontProperties(fname=fontfamily_text)
        opensans._size = font_size
        opensans._weight = font_weight
        print(f"custom_font: %s seconds ---" % (time.time() - start_time))

        return opensans

    @staticmethod
    def get_font_title():
        return FontHelper.custom_font(24, 'bold', 'Open Sans')

    @staticmethod
    def get_font_sub_title():
        return FontHelper.custom_font(14, 'bold', 'Monteserrat')
    #
    @staticmethod# Adjust the cache size as per your needs
    @functools.lru_cache(maxsize=128)
    def cache_image(font_size=12, font_weight='normal', font_name='Open Sans'):
        return FontHelper.custom_font(font_size=font_size, font_weight=font_weight, font_name=font_name)
    def get_custom_font(self, font_size, font_weight, font_name='Open Sans'):
    #
         if font_name == 'Open Sans':
             custom_font = self.sub_text_font
    
         else:
             custom_font = self.title_font
    
         custom_font._size = font_size
         custom_font._weight = font_weight
         return custom_font