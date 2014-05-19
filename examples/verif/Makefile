OBJS:=verif_train.prototxt verif_test.prototxt verif_feature.prototxt

all: $(OBJS)

%.prototxt: %.prototxt.partial verif_net.template
	cpp $< > $@

verif_test.prototxt: verif_test.prototxt.partial verif_net.template
	cpp $< |grep -v blobs_lr > $@

clean:
	rm -f $(OBJS)
