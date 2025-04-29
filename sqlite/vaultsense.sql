CREATE TABLE images (
    image_id     INTEGER PRIMARY KEY,
    bytes        BLOB            NOT NULL,
    format       TEXT            NOT NULL,
    width        INTEGER         NOT NULL,
    height       INTEGER         NOT NULL,
    created_ts   DATETIME        DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE milvus_map (
    vector_id   BIGINT PRIMARY KEY,
    image_id    INTEGER REFERENCES images(image_id)
);