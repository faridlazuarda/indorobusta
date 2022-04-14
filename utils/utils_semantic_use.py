import numpy as np
import jax.numpy as jnp
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tensorflow_hub as hub

class USE(object):
    def __init__(self):
        import tensorflow_hub as hub
        super(USE, self).__init__()
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
        self.embed = hub.load(module_url)
        # print(tf.executing_eagerly())
        # self.embed = hub.load( DataManager.load("AttackAssist.UniversalSentenceEncoder") )

    def semantic_sim(self, sentA : str, sentB : str) -> float:
        """
        Args:
            sentA: The first sentence.
            sentB: The second sentence.

        Returns:
            Cosine distance between two sentences.
        
        """
        # ret = jnp.array(self.embed([sentA, sentB]).numpy())
        ret = self.embed([sentA, sentB]).numpy()
        # ic(ret)
        # .block_until_ready()  
        # return jnp.dot(ret[0], ret[1]).block_until_ready()  / (np.linalg.norm(ret[0]) * np.linalg.norm(ret[1]))
        return ret[0].dot(ret[1]) / (np.linalg.norm(ret[0]) * np.linalg.norm(ret[1]))

    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score(input["x"], adversarial_sample)