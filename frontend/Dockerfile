# Use the official Node.js image as the base image
FROM node:18-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install --force --legacy-peer-deps

# Copy the rest of the application code to the container
COPY . .

# Build the React app for production
RUN npm run build

# Install 'serve' to serve the production build
RUN npm install -g serve

# Expose port 3000
EXPOSE 3000

# Serve the build directory
CMD ["serve", "-s", "build", "-l", "3000"]
