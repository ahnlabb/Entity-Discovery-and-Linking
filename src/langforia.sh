#!/bin/bash
curl "http://vilde.cs.lth.se:9000/$1/corenlp_3.8.0/api/tsv" -H 'Accept-Encoding: gzip, deflate' -H 'Content-Type: text/plain; charset=UTF-8' -H 'Accept: */*' --data-binary "$2" --compressed
