from deep_translator import GoogleTranslator
import tensorflow as tf

# translator = GoogleTranslator(source="id", target="ms")

# print(translator.translate("kelebihan"))
# print(tf. __version__)

googletranslator = GoogleTranslator()

# default return type is a list
langs_list = googletranslator.get_supported_languages()  # output: [arabic, french, english etc...]
print(langs_list)

# alternatively, you can the dictionary containing languages mapped to their abbreviation
langs_dict = googletranslator.get_supported_languages(as_dict=True)  # output: {arabic: ar, french: fr, english:en etc...}
print(langs_dict)