import dotenv from 'dotenv';

dotenv.config();

const host_name = process.env.HOST_NAME || "localhost";
const url_prefix = host_name + ":8000/api"

export const settings = {
    url_prefix: url_prefix,
    api_url: url_prefix + "",
}