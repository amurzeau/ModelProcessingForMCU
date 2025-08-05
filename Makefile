
all: st_validate st ethos st_validate_flash st_flash

tmp/%_p.onnx: %.onnx decompose_rnn.py
	python3 decompose_rnn.py $<

tmp/%_q_fast.onnx: tmp/%_p.onnx quantize_model_onnx.py
	python3 quantize_model_onnx.py --fast $< $@

tmp/%_q_accurate.onnx: tmp/%_p.onnx quantize_model_onnx.py
	python3 quantize_model_onnx.py --accurate $< $@

tmp/%_validation.npz: tmp/%.onnx preprocessing_wav.py
	python3 preprocessing_wav.py $<

tmp/%_int8.tflite: tmp/%.onnx
	python3 quantize_model_tflite.py $<


tmp/denoiser_dns_p_int8_H256_vela.txt: tmp/denoiser_dns_p_int8.tflite ethos_generate.sh
	./ethos_generate.sh

ethos: tmp/denoiser_dns_p_int8_H256_vela.txt


tmp/%_st_analyze_report.txt: tmp/%.onnx st_generate.sh config/user_neuralart.json
	./st_generate.sh $<
	cp st_ai_output/$(basename $(notdir $<))_analyze_report.txt $@

st: tmp/denoiser_dns_q_accurate_st_analyze_report.txt tmp/denoiser_dns_q_fast_st_analyze_report.txt


tmp/%_st_flash_analyze_report.txt: tmp/%.onnx st_generate_flash.sh config/user_neuralart.json
	./st_generate_flash.sh $<
	cp st_ai_output/$(basename $(notdir $<))_flash_analyze_report.txt $@

st_flash: tmp/denoiser_dns_q_accurate_st_flash_analyze_report.txt tmp/denoiser_dns_q_fast_st_flash_analyze_report.txt


tmp/%_st_validate_report.txt: tmp/%.onnx tmp/denoiser_dns_p_validation.npz st_validate_true.sh config/user_neuralart.json
	./st_validate_true.sh $<
	cp st_ai_output/network_validate_report.txt $@

st_validate: tmp/denoiser_dns_q_accurate_st_validate_report.txt tmp/denoiser_dns_q_fast_st_validate_report.txt


tmp/%_st_flash_validate_report.txt: tmp/%.onnx tmp/denoiser_dns_p_validation.npz st_validate_true_flash.sh config/user_neuralart.json
	./st_validate_true_flash.sh $<
	cp st_ai_output/network_validate_report.txt $@

st_validate_flash: tmp/denoiser_dns_q_accurate_st_flash_validate_report.txt tmp/denoiser_dns_q_fast_st_flash_validate_report.txt

.PHONY: ethos st st_validate st_validate_flash
.SUFFIXES:
.NOTPARALLEL:
.PRECIOUS: %
