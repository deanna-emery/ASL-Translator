# American Sign Language Translation: An Approach Combining MoViNets and T5

### Running the model

- You may generate predictions using our HuggingFace API: https://huggingface.co/spaces/deanna-emery/ASL-MoViNet-T5-translator
    - To download the fine-tuned MoViNets model, you must clone the API repo from the link above.
    - The fine-tuned sentence-level T5 model can be found in HuggingFace at: https://huggingface.co/deanna-emery/ASL_t5_movinet_sentence
    - The word-level T5 model can be found at: https://huggingface.co/deanna-emery/ASL_t5_word_epoch15_1204

To run the model, you must first clone the tensorflow models repo to access the MoViNet source code: https://github.com/tensorflow/models/tree/master/official/projects/movinet

Then you must copy the contents of our `movinet_modifications` folder into the tensorflow models repo in the following location:
```
cp ASL-Translator/movinet_modifications/* models/official/projects/movinet/modeling/
```

Finally, you may load the models with the following code:

```
import tensorflow as tf, tf_keras
import tensorflow_hub as hub
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model_a2_modified as movinet_model_modified


movinet_path = 'movinet_checkpoints_a2_epoch9'
movinet_model = tf_keras.models.load_model(movinet_path)

tokenizer = AutoTokenizer.from_pretrained("t5-base")
t5_model = TFAutoModelForSeq2SeqLM.from_pretrained("deanna-emery/ASL_t5_word_epoch15_1204")
```


### About the model

For more information on the details of the model, please refer to the following [paper](https://github.com/deanna-emery/ASL-Translator/blob/9b000d39ef8d35c8334941c97d620005bd8c6f62/American_Sign_Language_Translation.pdf). You can also find a high-level summary [here](https://www.ischool.berkeley.edu/projects/2023/signsense-american-sign-language-translation).