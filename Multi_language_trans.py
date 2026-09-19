import torch
import gradio as gr
import json
from transformers import pipeline

#model_path = r"C:\Users\91738\.cache\huggingface\hub\models--facebook--nllb-200-distilled-600M\snapshots\f8d333a098d19b4fd9a8b18f94170487ad3f821d"
model_path = ("../Models/models--facebook--nllb-200-distilled-600M/snapshots/"
              "f8d333a098d19b4fd9a8b18f94170487ad3f821d")

pipe = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",

)
text_translator = pipeline("translation", model=model_path, torch_dtype=torch.bfloat16)


with open("Files/language.json", "r", encoding="utf-8") as f:
    language_data = json.load(f)


def get_FLORES_code_from_language(language):
    for entry in language_data:
        if entry['Language'].lower()== language.lower():
            return entry["FLORES-200 code"]
    return None


def translate_text(text, destination_language):
    # text = "Hello friends, How are you"
    dest_code = get_FLORES_code_from_language(destination_language)
    translation = text_translator(text, src_lang="eng_Latn", tgt_lang=dest_code)
    return translation[0]["translation_text"]



gr.close_all()

demo = gr.Interface(fn=translate_text,
                    inputs=[gr.Textbox(label="Input text to translate",lines=9), gr.Dropdown(["Acehnese (Arabic script)","Acehnese (Latin script)","Mesopotamian Arabic","Ta’izzi-Adeni Arabic",
                                                                                             "Tunisian Arabic","Afrikaans","South Levantine Arabic","Akan","Amharic","North Levantine Arabic","Modern Standard Arabic",
                                                                                             "Modern Standard Arabic (Romanized)","Najdi Arabic","Moroccan Arabic","Egyptian Arabic","Assamese","Asturian","Awadhi",
                                                                                             "Central Aymara","South Azerbaijani","North Azerbaijani","Bashkir","Bambara","Balinese","Belarusian","Bemba","Bengali",
                                                                                             "Bhojpuri","Banjar (Arabic script)","Banjar (Latin script)","Standard Tibetan","Bosnian","Buginese","Bulgarian","Catalan",
                                                                                             "Cebuano","Czech","Chokwe","Central Kurdish","Crimean Tatar","Welsh","Danish","German","Southwestern Dinka","Dyula",
                                                                                             "Dzongkha","Greek","English","Esperanto","Estonian","Basque","Ewe","Faroese","Fijian","Finnish","Fon","French","Friulian",
                                                                                             "Nigerian Fulfulde","Scottish Gaelic","Irish","Galician","Guarani","Gujarati","Haitian Creole","Hausa","Hebrew","Hindi",
                                                                                             "Chhattisgarhi","Croatian","Hungarian","Armenian","Igbo","Ilocano","Indonesian","Icelandic","Italian","Javanese","Japanese",
                                                                                             "Kabyle","Jingpho","Kamba","Kannada","Kashmiri (Arabic script)","Kashmiri (Devanagari script)","Georgian",
                                                                                             "Central Kanuri (Arabic script)","Central Kanuri (Latin script)","Kazakh","Kabiyè","Kabuverdianu","Khmer","Kikuyu",
                                                                                             "Kinyarwanda","Kyrgyz","Kimbundu","Northern Kurdish","Kikongo","Korean","Lao","Ligurian","Limburgish","Lingala",
                                                                                             "Lithuanian","Lombard","Latgalian","Luxembourgish","Luba-Kasai","Ganda","Luo","Mizo","Standard Latvian","Magahi","Maithili",
                                                                                             "Malayalam","Marathi","Minangkabau (Arabic script)","Minangkabau (Latin script)","Macedonian","Plateau Malagasy","Maltese",
                                                                                             "Meitei (Bengali script)","Halh Mongolian","Mossi","Maori","Burmese","Dutch","Norwegian Nynorsk","Norwegian Bokmål","Nepali",
                                                                                             "Northern Sotho","Nuer","Nyanja","Occitan","West Central Oromo","Odia","Pangasinan","Eastern Panjabi","Papiamento",
                                                                                             "Western Persian","Polish","Portuguese","Dari","Southern Pashto","Ayacucho Quechua","Romanian","Rundi","Russian","Sango",
                                                                                             "Sanskrit","Santali","Sicilian","Shan","Sinhala","Slovak","Slovenian","Samoan","Shona","Sindhi","Somali","Southern Sotho",
                                                                                             "Spanish","Tosk Albanian","Sardinian","Serbian","Swati","Sundanese","Swedish","Swahili","Silesian","Tamil","Tatar","Telugu",
                                                                                             "Tajik","Tagalog","Thai","Tigrinya","Tamasheq (Latin script)","Tamasheq (Tifinagh script)","Tok Pisin","Tswana","Tsonga",
                                                                                             "Turkmen","Tumbuka","Turkish","Twi","Central Atlas Tamazight","Uyghur","Ukrainian","Umbundu","Urdu","Northern Uzbek",
                                                                                             "Venetian","Vietnamese","Waray","Wolof","Xhosa","Eastern Yiddish","Yoruba","Yue Chinese","Chinese (Simplified)",
                                                                                             "Chinese (Traditional)","Standard Malay","Zulu"],label="Select Destination Language")],
                    outputs=[gr.Textbox(label="Translated Text",lines=9)],
                    title ="Multi Language Translator",
                    description="THIS APPLICATION WILL BE USED TO TRANSLATE ANY ENGLISH TEXT TO MULTILANGUAGES."
                    )

demo.launch()






