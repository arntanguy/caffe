OBJS:=verif_train.prototxt verif_test.prototxt

all: $(OBJS)

%.prototxt: %.prototxt.partial
	cpp $< > $@

verif_test.prototxt: verif_test.prototxt.partial
	cpp $< |grep -v blobs_lr > $@

%.prototxt: verif_net.template

clean:
	rm -f $(OBJS)
