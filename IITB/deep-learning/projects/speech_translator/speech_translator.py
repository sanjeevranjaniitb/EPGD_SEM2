# Environment used is anaconda-nlp
import sounddevice as sd
import numpy as np
import speech_recognition as sr
from googletrans import Translator
import sys # Import sys to exit the script if necessary

def record_audio(sample_rate, duration):
    """Records audio from the microphone for a specified duration with error handling.

    Args:
        sample_rate: The sample rate for the recording (e.g., 44100 Hz).
        duration: The duration of the recording in seconds.

    Returns:
        A NumPy array containing the recorded audio data, or None if an error occurs.
    """
    print("Recording started...")
    try:
        # Check for available audio devices before recording
        devices = sd.query_devices()
        if not devices:
            print("No audio devices found. Please check your microphone setup.")
            return None

        # You might want to add logic here to select a specific device if needed
        # For simplicity, sounddevice often picks a default input device

        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")
        return audio_data
    except sd.PortAudioError as e:
        print(f"Error accessing microphone: {e}")
        print("Please ensure your microphone is connected and permissions are granted.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during recording: {e}")
        return None

def recognize_speech(audio_data, sample_rate):
    """Converts audio data to text using speech recognition with error handling.

    Args:
        audio_data: A NumPy array containing the audio data.
        sample_rate: The sample rate of the audio data.

    Returns:
        The recognized text, or None if recognition fails.
    """
    if audio_data is None:
        return None

    r = sr.Recognizer()
    # Convert numpy array to AudioData
    # Assuming 2 bytes per sample for np.int16
    audio_source = sr.AudioData(audio_data.tobytes(), sample_rate, 2)

    try:
        print("Recognizing speech...")
        # Use Google Web Speech API for recognition
        text = r.recognize_google(audio_source)
        print(f"Recognized text: {text}")
        return text
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        print("Please check your internet connection and try again.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during speech recognition: {e}")
        return None


def translate_text(text, target_languages):
    """Translates text into multiple target languages with error handling.

    Args:
        text: The text to translate.
        target_languages: A list of target language codes (e.g., ['es', 'fr', 'de']).

    Returns:
        A tuple containing the original text and a dictionary of translations,
        where keys are language codes and values are translated text.
        Returns (None, None) if an error occurs during translation.
    """
    if text is None:
        return None, None

    translator = Translator()
    translations = {}
    try:
        print("Translating text...")
        for lang_code in target_languages:
            translation = translator.translate(text, dest=lang_code)
            translations[lang_code] = translation.text
        print("Translation finished.")
        return text, translations
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        print("Please check your internet connection and ensure the language code is valid.")
        return None, None

def display_translations(original_text, translated_texts):
    """Displays the original text and its translations.

    Args:
        original_text: The original text.
        translated_texts: A dictionary of translated texts, where keys are
                          language codes and values are translated text.
    """
    if original_text is None or translated_texts is None:
        print("No text or translations to display.")
        return

    print("\n--- Results ---")
    print(f"Original text: {original_text}")
    print("Translations:")
    if translated_texts:
        for lang_code, translated_text in translated_texts.items():
            print(f"{lang_code.upper()}: {translated_text}")
    else:
        print("No translations available.")
    print("---------------\n")


# --- Command-line Interface ---
if __name__ == "__main__":
    print("Welcome to the Speech Translator!")

    # Define available languages for translation
    available_languages = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'zh-cn': 'Chinese (Simplified)',
        'hi': 'Hindi',
        # Add more languages as needed
    }

    print("\nAvailable languages for translation:")
    for code, name in available_languages.items():
        print(f"  {code}: {name}")

    target_lang_code = input("\nEnter target language code (e.g., es, fr, de): ").lower()

    if target_lang_code not in available_languages:
        print(f"Invalid language code: {target_lang_code}. Please use one of the available codes.")
        sys.exit(1) # Exit the script if the language code is invalid

    sample_rate = 44100  # Standard sample rate
    duration = 5         # Duration for recording in seconds

    # --- Main process ---
    audio_data = record_audio(sample_rate, duration)

    if audio_data is not None:
        recognized_text = recognize_speech(audio_data, sample_rate)

        if recognized_text is not None:
            # Translate to the selected language and potentially English if it's not the target
            target_languages = [target_lang_code]
            if target_lang_code != 'en' and 'en' not in target_languages:
                 target_languages.append('en') # Add English as a reference

            original, translations = translate_text(recognized_text, target_languages)

            display_translations(original, translations)
        else:
            print("Could not recognize speech. Please try again.")
    else:
        print("Audio recording failed. Cannot proceed with recognition and translation.")

    print("Script finished.")