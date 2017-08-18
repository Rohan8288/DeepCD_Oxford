# DeepCD

  You can download the pre-extracted patches and the pre-extracted features from https://drive.google.com/open?id=0B7c9IgGVCg6IVmUtY01Ga3hqWE0
  
  You can download the pre-trained models from https://drive.google.com/open?id=0B7c9IgGVCg6INGp5OW5CcVhCTFU

# usage

1. dataProcess.m : extract patches from the Oxford dataset by VLFeat
    
    or you can directly use the pre-extracted patches

<pre>
    dataProcess (run in matlab)
</pre>
    
2. extract.sh : extract features from patches by CNN-based models

    or you can directly use the pre-extracted features
    
<pre>
    bash extract.sh
</pre>
    
3. result.m : evaluate the features extractes by CNN-based models

<pre>
    result (run in matlab)
</pre>
