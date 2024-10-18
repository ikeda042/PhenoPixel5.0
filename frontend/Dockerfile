# Use the official Node.js image as the base image
FROM node:18-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies using legacy peer deps flag to ignore conflicts
RUN npm install --force --legacy-peer-deps

# Globally install react-scripts and typescript
RUN npm install react-scripts --force --legacy-peer-deps

# Copy the rest of the application code to the container
COPY . .

# Ensure TypeScript is installed (redundant but explicit)
RUN npm install typescript --force --legacy-peer-deps

# Build the React app for production
RUN npm run build

# Expose port 3000 to the outside world
EXPOSE 3000

# Start the React app
CMD ["npm", "start"]
