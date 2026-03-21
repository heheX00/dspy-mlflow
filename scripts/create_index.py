from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
index_name = "gkg_sandbox"

mapping = {
    "mappings": {
        "properties": {
            "@timestamp": {"type": "keyword"},
            "@version": {"type": "keyword"},

            # Explicitly define the input format so ES does not guess incorrectly
            "GkgRecordId.Date": {
                "type": "date",
                "format": "strict_date_optional_time||yyyyMMddHHmmss||epoch_millis"
            },
            "GkgRecordId.NumberInBatch": {"type": "integer"},
            "GkgRecordId.Translingual": {"type": "boolean"},

            "RecordId": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},

            # Store V21Date as a real ISO date string at ingest time
            "V21Date": {
                "type": "date",
                "format": "strict_date_optional_time||yyyyMMddHHmmss||epoch_millis"
            },

            "V15Tone.ActivityRefDensity": {"type": "float"},
            "V15Tone.NegativeScore": {"type": "float"},
            "V15Tone.Polarity": {"type": "float"},
            "V15Tone.PositiveScore": {"type": "float"},
            "V15Tone.SelfGroupRefDensity": {"type": "float"},
            "V15Tone.Tone": {"type": "float"},

            "V21AllNames.Name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "V21Quotations.Quote": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "V21Quotations.Verb": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},

            "V21RelImg": {"type": "keyword"},
            "V21ShareImg": {"type": "keyword"},
            "V21SocImage": {"type": "keyword"},
            "V21SocVideo.V21SocVideo": {"type": "keyword"},

            "V2DocId": {"type": "keyword"},

            "V2EnhancedThemes.V2Theme": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}}
            },

            "V2ExtrasXML.AltUrl": {"type": "keyword"},
            "V2ExtrasXML.AltUrlAmp": {"type": "keyword"},
            "V2ExtrasXML.Author": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "V2ExtrasXML.Links": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "V2ExtrasXML.PubTimestamp": {"type": "date"},
            "V2ExtrasXML.Title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},

            "V2GCAM.DictionaryDimId": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}}
            },

            "V2Locations.ADM1Code": {"type": "keyword"},
            "V2Locations.ADM2Code": {"type": "keyword"},
            "V2Locations.CountryCode": {"type": "keyword"},
            "V2Locations.FeatureId": {"type": "keyword"},
            "V2Locations.FullName": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "V2Locations.LocationLatitude": {"type": "double"},
            "V2Locations.LocationLongitude": {"type": "double"},

            "V2Orgs.V1Org": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "V2Persons.V1Person": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},

            "V2SrcCmnName.V2SrcCmnName": {"type": "keyword"},
            "V2SrcCollectionId.V2SrcCollectionId": {"type": "keyword"},

            "datatype": {"type": "keyword"},
            "event.original": {"type": "text"},
            "filename": {"type": "keyword"},
            "filename_path": {"type": "keyword"},
            "host.name": {"type": "keyword"},
            "location": {"type": "geo_point"},
            "log.file.path": {"type": "keyword"}
        }
    }
}

if es.indices.exists(index=index_name):
    print(f"Index '{index_name}' already exists")
else:
    es.indices.create(index=index_name, body=mapping)
    print(f"Created index '{index_name}'")