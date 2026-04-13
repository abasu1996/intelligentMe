targetScope = 'subscription'

@description('AZD environment name.')
param environmentName string

@description('Azure region for all resources.')
param location string

@description('Required Ariba realm name.')
param aribaRealm string

@secure()
@description('Required Ariba OAuth client ID.')
param aribaClientId string

@secure()
@description('Required Ariba OAuth client secret.')
param aribaClientSecret string

@secure()
@description('Required Ariba API key.')
param aribaApiKey string

@description('Optional Ariba OAuth base URL override.')
param aribaOauthUrl string = 'https://api.ariba.com'

@description('Optional Ariba API base URL override.')
param aribaApiUrl string = 'https://openapi.ariba.com/api'

var tags = {
  'azd-env-name': environmentName
  project: 'ariba-mcp'
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
    name: 'ariba'
    serviceName: 'ariba'
    location: location
    tags: tags
    aribaRealm: aribaRealm
    aribaClientId: aribaClientId
    aribaClientSecret: aribaClientSecret
    aribaApiKey: aribaApiKey
    aribaOauthUrl: aribaOauthUrl
    aribaApiUrl: aribaApiUrl
  }
}

output AZURE_RESOURCE_GROUP string = resourceGroup.name
output API_URL string = appService.outputs.apiUrl
output APPLICATIONINSIGHTS_CONNECTION_STRING string = appService.outputs.applicationInsightsConnectionString
