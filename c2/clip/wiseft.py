from c2.clip.linearProbe import LinearProbe


class WiSEFT(LinearProbe):
    name = "wiseft_clip"

    def __init__(self, class_names, mix=0.5, **hparams):
        super().__init__(class_names, mix=mix, **hparams)