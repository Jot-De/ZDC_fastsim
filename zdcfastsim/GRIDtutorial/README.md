# Instruction how to submit to the GRID in easy way

The code in this directory can be used to
a) show how to submit a local script as an ALICE GRID job
b) show a script that produces ZDC training data for pp collisions and which produces
     training images in text form (script.sh)

One can run this script on the GRID with an O2 software tag of nightly-20230427-1, a name of ZDCtest1
and a timeout of 1h like so

```
./grid_submit.sh --script script.sh --o2tag nightly-20230427-1 --jobname ZDCtest1 --ttl 3600
```

Result files will be available under your AliEn folder + 'selftest' + ZDCtest1...