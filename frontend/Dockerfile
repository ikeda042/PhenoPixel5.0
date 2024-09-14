# Use the official Node.js image as the base image
FROM node:18-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies using legacy peer deps flag to ignore conflicts
RUN npm install --legacy-peer-deps

RUN npm install react-script
RUN npm install typescript

# Copy the rest of the application code to the container
COPY . .

# Build the React app for production
RUN npm run build

# Expose port 3000 to the outside world
EXPOSE 3000

# Start the React app
CMD ["npm", "start"]