import numpy as np
import tensorflow
import tensorflow.compat.v1 as tf
# import jax.numpy as jnp
tf.disable_v2_behavior()
tf.disable_eager_execution()
import tensorflow_hub as hub

class USE(object):
    def __init__(self, module_url=None):
        super(USE, self).__init__()
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
        self.embed = hub.load(module_url)
        
        def embed(input):
            return model(input)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def semantic_sim(self, sents1, sents2):
        # with tf.Session() as session:
        #     session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        #     with tf.device('GPU'):
        #         message_embeddings_ = session.run(self.embed([sents1, sents2]))
        
        if sents1.lower() == sents2.lower():
            return 1.000
        
        message_embeddings_ = self.embed([sents1, sents2])
        message_embeddings_ = message_embeddings_.eval(session=self.sess)
        corr = np.tensordot(message_embeddings_, message_embeddings_, axes=(-1,-1))
        
        if corr[0][1] > 1:
            return 1.000
        return corr[0][1]