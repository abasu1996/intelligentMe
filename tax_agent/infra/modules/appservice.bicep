targetScope = 'resourceGroup'

@description('Base resource name.')
param name string

@description('Deployment location.')
param location string = resourceGroup().location

@description('Tags applied to all resources.')
param tags object = {}

@description('AZD service name used for deployment tagging.')
param serviceName string

var resourceSuffix = take(uniqueString(subscription().id, resourceGroup().name, name), 6)
var appServicePlanName = 'asp-${name}-${resourceSuffix}'
var webAppName = 'app-${name}-${resourceSuffix}'
var logAnalyticsName = 'log-${name}-${resourceSuffix}'
var appInsightsName = 'appi-${name}-${resourceSuffix}'

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  tags: tags
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspace.id
  }
}

resource appServicePlan 'Microsoft.Web/serverfarms@2022-09-01' = {
  name: appServicePlanName
  location: location
  kind: 'linux'
  sku: {
    name: 'F1'
    tier: 'Free'
  }
  tags: tags
  properties: {
    reserved: true
  }
}

resource webApp 'Microsoft.Web/sites@2022-09-01' = {
  name: webAppName
  location: location
  kind: 'app,linux'
  tags: union(tags, {
    'azd-service-name': serviceName
  })
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    clientAffinityEnabled: false
    publicNetworkAccess: 'Enabled'
    serverFarmId: appServicePlan.id
    httpsOnly: true
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.12'
      minTlsVersion: '1.2'
      ftpsState: 'Disabled'
      http20Enabled: true
      healthCheckPath: '/health'
      appCommandLine: 'sh startup.sh'
    }
  }
}

resource webAppAppSettings 'Microsoft.Web/sites/config@2022-09-01' = {
  name: 'appsettings'
  parent: webApp
  properties: {
    SCM_DO_BUILD_DURING_DEPLOYMENT: 'true'
    APPLICATIONINSIGHTS_CONNECTION_STRING: applicationInsights.properties.ConnectionString
    APPINSIGHTS_INSTRUMENTATIONKEY: applicationInsights.properties.InstrumentationKey
  }
}

output apiUrl string = 'https://${webApp.properties.defaultHostName}'
output applicationInsightsConnectionString string = applicationInsights.properties.ConnectionString
