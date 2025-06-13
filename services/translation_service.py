<<<<<<< HEAD
from googletrans import Translator, LANGUAGES

class TranslationService:
    def __init__(self):
        self.translator = Translator()
        self.supported_languages = LANGUAGES
        self.name_to_code = {name.lower(): code for code, name in LANGUAGES.items()}

    def get_supported_languages(self):
        """Returns a dictionary of supported language codes and their names."""
        return self.supported_languages

    def is_language_supported(self, lang):
        """Check if the language is supported (by code or name)."""
        lang = lang.lower()
        return lang in self.supported_languages or lang in self.name_to_code

    def _resolve_language_code(self, lang):
        """Convert language name to code if necessary."""
        lang = lang.lower()
        if lang in self.supported_languages:  # it's already a code
            return lang
        elif lang in self.name_to_code:
            return self.name_to_code[lang]
        else:
            raise ValueError(f"Unsupported language: {lang}")

    def translate_text(self, text, dest_language='hi'):
        """
        Translate text to the specified language.
        
        Args:
            text (str): The original English text.
            dest_language (str): Language code or name (e.g., 'hi' or 'Hindi').
        
        Returns:
            dict: Contains 'translated_text', 'dest_language', 'language_name'
        """
        lang_code = self._resolve_language_code(dest_language)
        result = self.translator.translate(text, dest=lang_code)

        return {
            "translated_text": result.text,
            "dest_language": lang_code,
            "language_name": self.supported_languages[lang_code]
        }
=======
from googletrans import Translator, LANGUAGES

class TranslationService:
    def __init__(self):
        self.translator = Translator()
        self.supported_languages = LANGUAGES
        self.name_to_code = {name.lower(): code for code, name in LANGUAGES.items()}

    def get_supported_languages(self):
        """Returns a dictionary of supported language codes and their names."""
        return self.supported_languages

    def is_language_supported(self, lang):
        """Check if the language is supported (by code or name)."""
        lang = lang.lower()
        return lang in self.supported_languages or lang in self.name_to_code

    def _resolve_language_code(self, lang):
        """Convert language name to code if necessary."""
        lang = lang.lower()
        if lang in self.supported_languages:  # it's already a code
            return lang
        elif lang in self.name_to_code:
            return self.name_to_code[lang]
        else:
            raise ValueError(f"Unsupported language: {lang}")

    def translate_text(self, text, dest_language='hi'):
        """
        Translate text to the specified language.
        
        Args:
            text (str): The original English text.
            dest_language (str): Language code or name (e.g., 'hi' or 'Hindi').
        
        Returns:
            dict: Contains 'translated_text', 'dest_language', 'language_name'
        """
        lang_code = self._resolve_language_code(dest_language)
        result = self.translator.translate(text, dest=lang_code)

        return {
            "translated_text": result.text,
            "dest_language": lang_code,
            "language_name": self.supported_languages[lang_code]
        }
>>>>>>> a9174a78eb2841517af3526f031d0039f61cc97d
