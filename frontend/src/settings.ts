const host_name = process.env.HOST_NAME || "localhost";
const url_prefix = `http://${host_name}:8000/api`;

export const settings = {
    url_prefix: url_prefix,
    api_url: url_prefix + "",
};