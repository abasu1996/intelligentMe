targetScope = 'subscription'

@description('AZD environment name.')
param environmentName string

@description('Azure region for all resources.')
param location string

var tags = {
  'azd-env-name': environmentName
  project: 'tax-agent'
}

resource resourceGroup 'Microsoft.Resources/resourceGroups@2023-07-01' = {
  name: 'rg-${environmentName}'
  location: location
  tags: tags
}

module appService './modules/appservice.bicep' = {
  name: 'appservice'
  scope: resourceGroup
  params: {
    name: 'taxagent'
    location: location
    tags: tags
    serviceName: 'taxagent'
  }
}

output AZURE_RESOURCE_GROUP string = resourceGroup.name
output API_URL string = appService.outputs.apiUrl
output APPLICATIONINSIGHTS_CONNECTION_STRING string = appService.outputs.applicationInsightsConnectionString
